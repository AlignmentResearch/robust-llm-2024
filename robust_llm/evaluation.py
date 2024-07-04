"""Code for evaluating an attack on a single model. Used by multiple pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import wandb

from robust_llm import logger
from robust_llm.attacks.attack import Attack
from robust_llm.defenses.defense import FilteringDefendedModel
from robust_llm.defenses.perplexity import PerplexityDefendedModel
from robust_llm.evaluation_utils import (
    AttackResults,
    assert_same_data_between_processes,
)
from robust_llm.logging_utils import WandbTable
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks import BinaryCallback, CallbackInput


def do_adversarial_evaluation(
    victim: WrappedModel,
    dataset: RLLMDataset,
    attack: Attack,
    final_success_binary_callback: BinaryCallback,
    num_examples_to_log_detailed_info: Optional[int],
    adv_training_round: int,
    victim_training_step_count: int,
    victim_training_datapoint_count: int,
    global_step_count: int,
    global_datapoint_count: int,
    wandb_table: Optional[WandbTable] = None,
) -> dict[str, float]:
    """Performs adversarial evaluation and logs the results."""
    # If a wandb table was passed in, we write logging information to that table, and
    # do not take responsibility for writing the table to disk. If no table was
    # passed in, we will create our own and save it to disk.
    save_table: bool = wandb_table is None
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

    callback_input = CallbackInput(
        # TODO(ian): Work out where to apply chat template.
        input_data=victim.maybe_apply_chat_template(dataset.ds["text"]),
        clf_label_data=dataset.ds["clf_label"],
        gen_target_data=dataset.ds["gen_target"],
    )
    pre_attack_out = final_success_binary_callback(
        victim,
        callback_input,
    )
    pre_attack_out.maybe_log_info("pre_attack_callback")
    pre_attack_successes = pre_attack_out.successes

    # Reduce the dataset to only the examples that the model got correct, since
    # we only compute attack metrics on those anyway.
    indices_to_attack = [i for i, success in enumerate(pre_attack_successes) if success]
    if len(indices_to_attack) == 0:
        raise ValueError("No examples to attack in adversarial evaluation!")
    dataset_to_attack = dataset.get_subset(indices_to_attack)

    attacked_dataset, info_dict = attack.get_attacked_dataset(dataset=dataset_to_attack)

    # In case the attack changed the victim from eval() mode, we set it again here.
    victim.eval()

    # We use dataset_to_attack so that we use the same examples as in attacked_dataset
    original_input_data = victim.maybe_apply_chat_template(dataset_to_attack.ds["text"])

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
    post_attack_out.maybe_log_info("post_attack_callback")
    post_attack_successes = post_attack_out.successes

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
    assert len(set(metrics.keys()) & set(info_dict.keys())) == 0
    metrics |= info_dict

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

    if (
        num_examples_to_log_detailed_info is not None
        and victim.accelerator.is_main_process
    ):
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
        )

    if victim.accelerator.is_main_process:
        wandb.log(metrics, commit=True)
        logger.info("Adversarial evaluation metrics:")
        logger.info(metrics)
        wandb_table = (
            WandbTable("adversarial_eval/table") if wandb_table is None else wandb_table
        )
        wandb_table.add_data(metrics)
        if save_table:
            # Since no table was passed in, we must save it here to keep the data.
            wandb_table.save()

    return metrics


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
        indices_to_attack: The indices in the pre-attack datasetof the examples
            that were attacked.
        num_examples_to_log_detailed_info: The number of examples to log.
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
    )
    # We only log examples from those that were actually attacked.
    indices_to_record = indices_to_attack[:num_examples_to_log_detailed_info]
    for post_attack_idx, pre_attack_idx in enumerate(indices_to_record):
        table.add_data(
            original_texts[pre_attack_idx],
            original_labels[pre_attack_idx],
            original_successes[pre_attack_idx],
            original_flags[pre_attack_idx],
            attacked_texts[post_attack_idx],
            attacked_labels[post_attack_idx],
            attacked_successes[post_attack_idx],
            attacked_flags[post_attack_idx],
        )

    wandb.log({"adversarial_eval/examples": table}, commit=False)
