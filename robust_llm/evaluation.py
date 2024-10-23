"""Code for evaluating an attack on a single model. Used by multiple pipelines."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import wandb

from robust_llm import logger
from robust_llm.attacks.attack import Attack, AttackOutput
from robust_llm.defenses.defense import FilteringDefendedModel
from robust_llm.defenses.perplexity import PerplexityDefendedModel
from robust_llm.dist_utils import is_main_process
from robust_llm.evaluation_utils import (
    AttackResults,
    assert_same_data_between_processes,
)
from robust_llm.file_utils import ATTACK_DATA_NAME
from robust_llm.logging_utils import WandbTable, wandb_log
from robust_llm.metrics import maybe_compute_robustness_metrics
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks import BinaryCallback, CallbackInput
from robust_llm.scoring_callbacks.scoring_callback_utils import BinaryCallbackOutput
from robust_llm.utils import print_time


@print_time()
def do_adversarial_evaluation(
    victim: WrappedModel,
    dataset: RLLMDataset,
    attack: Attack,
    n_its: int,
    final_success_binary_callback: BinaryCallback,
    num_examples_to_log_detailed_info: Optional[int],
    adv_training_round: int,
    victim_training_step_count: int,
    victim_training_datapoint_count: int,
    global_step_count: int,
    global_datapoint_count: int,
    local_files_path: Path | None,
    wandb_table: Optional[WandbTable] = None,
    resume_from_checkpoint: bool = True,
    compute_robustness_metric: bool = True,
    upload_artifacts: bool = True,
) -> dict[str, float]:
    """Performs adversarial evaluation and logs the results."""
    wandb_table_exists = wandb_table is not None
    # We only commit if there is not a wandb_table, since a table implies that this
    # function is being called from the adversarial training pipeline.
    should_commit = not wandb_table_exists

    if victim.accelerator is None:
        raise ValueError("Accelerator must be provided")
    # Sanity check in case of a distributed run (with accelerate): check if every
    # process has the same dataset.
    # TODO(michal): Look into datasets code and make sure the dataset creation is
    # deterministic given seeds; this is especially important when using accelerate.
    assert_same_data_between_processes(victim.accelerator, dataset.ds["text"])
    assert_same_data_between_processes(victim.accelerator, dataset.ds["clf_label"])

    victim.eval()
    model_size = victim.n_params
    model_family = victim.family

    pre_attack_out = pre_attack_evaluation(
        victim=victim,
        dataset=dataset,
        final_success_binary_callback=final_success_binary_callback,
        should_commit=should_commit,
    )
    pre_attack_successes = pre_attack_out.successes

    # Reduce the dataset to only the examples that the model got correct, since
    # we only compute attack metrics on those anyway.
    indices_to_attack = [i for i, success in enumerate(pre_attack_successes) if success]
    if len(indices_to_attack) == 0:
        raise ValueError("No examples to attack in adversarial evaluation!")
    dataset_to_attack = dataset.get_subset(indices_to_attack)

    attack_out, attack_flops = attack_dataset(
        victim=victim,
        dataset_to_attack=dataset_to_attack,
        attack=attack,
        n_its=n_its,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    attacked_dataset = attack_out.dataset
    maybe_save_attack_data(
        victim,
        attack_out,
        should_commit,
        indices_to_attack,
        local_files_path,
        upload_artifacts=upload_artifacts,
    )
    # In case the attack changed the victim from eval() mode, we set it again here.
    victim.eval()

    # We use dataset_to_attack so that we use the same examples as in attacked_dataset
    original_input_data = victim.maybe_apply_user_template(dataset_to_attack.ds["text"])

    post_attack_out = post_attack_evaluation(
        victim=victim,
        attacked_dataset=attacked_dataset,
        final_success_binary_callback=final_success_binary_callback,
        original_input_data=original_input_data,
        should_commit=should_commit,
    )

    robustness_metric = maybe_compute_robustness_metrics(
        compute_robustness_metric=compute_robustness_metric,
        attack_out=attack_out,
        success_callback=final_success_binary_callback,
        model=victim,
    )

    post_attack_successes = post_attack_out.successes
    attack_success_rate = post_attack_successes.count(False) / len(
        post_attack_successes
    )
    logger.info(f"Attack success rate: {attack_success_rate}")

    attack_results = AttackResults(
        pre_attack_successes=pre_attack_successes,
        post_attack_successes=post_attack_successes,
    )
    pre_attack_flags: Sequence[bool] | Sequence[None]
    post_attack_flags: Sequence[bool] | Sequence[None]
    if isinstance(victim, FilteringDefendedModel):
        pre_attack_flags = victim.filter(dataset.ds["text"])
        post_attack_flags = victim.filter(attacked_dataset.ds["attacked_text"])
        attack_results = attack_results.with_defense_flags(
            pre_attack_flag_values=pre_attack_flags,
            post_attack_flag_values=post_attack_flags,
        )
    # If the model is not a FilteringDefendedModel, we set the flags to None for
    # logging.
    else:
        pre_attack_flags = [None] * len(pre_attack_successes)
        post_attack_flags = [None] * len(post_attack_successes)

    metrics = attack_results.compute_adversarial_evaluation_metrics()
    assert len(set(metrics.keys()) & set(attack_out.global_info.keys())) == 0
    metrics |= attack_out.global_info

    metrics |= _maybe_record_defense_specific_metrics(
        model=victim, dataset=dataset, attacked_dataset=attacked_dataset
    )
    metrics["adv_training_round"] = adv_training_round
    metrics["victim_training_step_count"] = victim_training_step_count
    metrics["victim_training_datapoint_count"] = victim_training_datapoint_count
    metrics["global_step_count"] = global_step_count
    metrics["global_datapoint_count"] = global_datapoint_count
    metrics["model_size"] = model_size
    metrics["model_family"] = model_family
    metrics["attack_flops"] = attack_flops
    metrics["flops_per_iteration"] = attack_flops / n_its

    if robustness_metric is not None:
        metrics |= robustness_metric.unwrap_metrics()
        robustness_metric.export_wandb_table()

    if num_examples_to_log_detailed_info is not None and is_main_process():
        _log_examples_to_wandb(
            original_texts=dataset.ds["text"],
            original_labels=dataset.ds["clf_label"],
            original_successes=pre_attack_successes,
            original_flags=pre_attack_flags,
            attacked_texts=attacked_dataset.ds["attacked_text"],
            attacked_labels=attacked_dataset.ds["attacked_clf_label"],
            attacked_successes=post_attack_successes,
            attacked_flags=post_attack_flags,
            indices_to_attack=indices_to_attack,
            num_examples_to_log_detailed_info=num_examples_to_log_detailed_info,
            **attack_out.per_example_info,
        )

    maybe_log_adversarial_eval_table(
        victim=victim,
        metrics=metrics,
        wandb_table=wandb_table,
        wandb_table_exists=wandb_table_exists,
    )

    return metrics


@print_time()
def pre_attack_evaluation(
    victim: WrappedModel,
    dataset: RLLMDataset,
    final_success_binary_callback: BinaryCallback,
    should_commit: bool,
) -> BinaryCallbackOutput:
    callback_input = CallbackInput(
        # TODO(ian): Work out where to apply chat template.
        input_data=victim.maybe_apply_user_template(dataset.ds["text"]),
        clf_label_data=dataset.ds["clf_label"],
        gen_target_data=dataset.ds["gen_target"],
    )
    pre_attack_out = final_success_binary_callback(
        victim,
        callback_input,
    )
    pre_attack_out.maybe_log_info("pre_attack_callback", commit=should_commit)
    return pre_attack_out


@print_time()
def attack_dataset(
    victim: WrappedModel,
    dataset_to_attack: RLLMDataset,
    attack: Attack,
    n_its: int,
    resume_from_checkpoint: bool,
) -> tuple[AttackOutput, int]:
    with victim.flop_count_context() as attack_flops:
        attack_out = attack.get_attacked_dataset(
            dataset=dataset_to_attack,
            n_its=n_its,
            # Only resume from checkpoint if we're in the evaluation pipeline as
            # otherwise we will be incorrectly reusing data in the case of adversarial
            # training.
            resume_from_checkpoint=resume_from_checkpoint,
        )
    return attack_out, attack_flops.flops


@print_time()
def maybe_save_attack_data(
    victim: WrappedModel,
    attack_out: AttackOutput,
    should_commit: bool,
    indices_to_attack: Sequence[int],
    local_files_path: Path | None,
    upload_artifacts: bool = True,
):
    """Saves the attack data to wandb if we're in the evaluation pipeline.

    Args:
        victim: The model that was attacked.
        attack_out: The AttackOutput object from the attack.
        should_commit: Whether to commit the data to wandb.
        indices_to_attack: The indices in the pre-attack dataset of the examples
            that were attacked. This is important for getting the correct labels
            and other information when loading from wandb.
        local_files_path: The path to the local files directory.
        upload_artifacts: Whether to upload the artifact to wandb.
    """
    assert victim.accelerator is not None
    # Only save attack data if we're in the evaluation pipeline - we don't care
    # about computing the robustness metric in adv training.
    if should_commit and is_main_process() and attack_out.attack_data is not None:
        assert isinstance(local_files_path, Path)
        dfs = attack_out.attack_data.to_dfs()
        for example_idx, df in zip(indices_to_attack, dfs, strict=True):
            df["example_idx"] = example_idx
        concat_df = pd.concat(dfs, ignore_index=True)
        local_files_path.mkdir(parents=True, exist_ok=True)
        path = str(local_files_path / ATTACK_DATA_NAME)
        concat_df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\")
        print(f"Saved attack data to {path}")
        if upload_artifacts:
            run = wandb.run
            assert run is not None
            artifact = wandb.Artifact(
                name=f"run-{run.id}-attack_data", type="attack_data"
            )
            artifact.add_file(path)
            artifact.save()


@print_time()
def post_attack_evaluation(
    victim: WrappedModel,
    attacked_dataset: RLLMDataset,
    final_success_binary_callback: BinaryCallback,
    original_input_data: Sequence[str],
    should_commit: bool,
) -> BinaryCallbackOutput:
    callback_input = CallbackInput(
        # NOTE: We don't apply the chat template here because we assume that the
        # attack already did that.
        # TODO(ian): Work out where to apply chat template.
        input_data=attacked_dataset.ds["attacked_text"],
        original_input_data=original_input_data,
        clf_label_data=attacked_dataset.ds["attacked_clf_label"],
        gen_target_data=attacked_dataset.ds["attacked_gen_target"],
    )
    post_attack_out = final_success_binary_callback(
        victim,
        callback_input,
    )
    post_attack_out.maybe_log_info("post_attack_callback", commit=should_commit)
    return post_attack_out


def _maybe_record_defense_specific_metrics(
    model: WrappedModel, dataset: RLLMDataset, attacked_dataset: RLLMDataset
) -> dict[str, Any]:

    metrics: dict[str, Any] = {}

    if isinstance(model, PerplexityDefendedModel) and model.cfg.save_perplexity_curves:
        # Get the approximate perplexities of the decoder on both
        # the original and attacked datasets.
        original_perplexities = model.get_all_perplexity_thresholds(
            dataset=dataset,
            text_column_to_use="text",
        )
        attacked_perplexities = model.get_all_perplexity_thresholds(
            dataset=attacked_dataset,
            text_column_to_use="attacked_text",
        )

        metrics["perplexity/decoder_perplexities_original"] = original_perplexities
        metrics["perplexity/decoder_perplexities_attacked"] = attacked_perplexities

    return metrics


@print_time()
def _log_examples_to_wandb(
    original_texts: Sequence[str],
    original_labels: Sequence[int],
    original_successes: Sequence[bool],
    original_flags: Sequence[bool] | Sequence[None],
    attacked_texts: Sequence[str],
    attacked_labels: Sequence[int],
    attacked_successes: Sequence[bool],
    attacked_flags: Sequence[bool] | Sequence[None],
    indices_to_attack: Sequence[int],
    num_examples_to_log_detailed_info: int,
    **other_attack_info,
) -> None:
    """Logs examples to wandb.

    Args:
        original_texts: The original texts.
        original_labels: The original labels.
        original_successes: The original successes.
        original_flags: The original defense flags.
        attacked_texts: The attacked texts.
        attacked_labels: The attacked labels.
        attacked_successes: The attacked successes.
        attacked_flags: The attacked defense flags.
        indices_to_attack: The indices in the pre-attack dataset of the examples
            that were attacked.
        num_examples_to_log_detailed_info: The number of examples to log.
        other_attack_info: Other attack-specific information to log.
            Should be dictionaries of str -> Sequence, where the sequence has
            the same length as the attacked_texts.
    """

    table = wandb.Table(
        columns=[
            "original_text",
            "original_label",
            "original_success",
            "original_flag",
            "attacked_text",
            "attacked_label",
            "attacked_success",
            "attacked_flag",
        ]
        + list(other_attack_info.keys())
    )
    # We only log examples from those that were actually attacked.
    indices_to_record = indices_to_attack[:num_examples_to_log_detailed_info]
    for post_attack_idx, pre_attack_idx in enumerate(indices_to_record):
        table.add_data(
            *[
                original_texts[pre_attack_idx],
                original_labels[pre_attack_idx],
                original_successes[pre_attack_idx],
                original_flags[pre_attack_idx],
                attacked_texts[post_attack_idx],
                attacked_labels[post_attack_idx],
                attacked_successes[post_attack_idx],
                attacked_flags[post_attack_idx],
            ]
            + [other_attack_info[key][post_attack_idx] for key in other_attack_info]
        )

    wandb_log({"adversarial_eval/examples": table}, commit=False)


@print_time()
def maybe_log_adversarial_eval_table(
    victim: WrappedModel,
    metrics: dict[str, Any],
    wandb_table: Optional[WandbTable] = None,
    wandb_table_exists: bool = False,
):
    assert victim.accelerator is not None
    if is_main_process():
        wandb_log(metrics, commit=True)
        logger.info("Adversarial evaluation metrics:")
        logger.info(metrics)
        wandb_table = (
            WandbTable("adversarial_eval/table") if wandb_table is None else wandb_table
        )
        wandb_table.add_data(metrics)
        if not wandb_table_exists:
            # If the wandb table already exists, we don't take responsibility for saving
            wandb_table.save()
