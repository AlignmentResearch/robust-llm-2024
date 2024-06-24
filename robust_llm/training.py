import dataclasses
import json
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import evaluate
import numpy as np
import torch
import transformers
import wandb
import wandb.util
from datasets import Dataset
from transformers import EvalPrediction, TrainingArguments
from transformers.trainer import (
    CONFIG_NAME,
    OPTIMIZER_NAME,
    SAFE_WEIGHTS_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
)
from typing_extensions import override

from robust_llm import logger
from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.callbacks import CustomLoggingWandbCallback
from robust_llm.config.configs import (
    AttackConfig,
    EnvironmentConfig,
    EvaluationConfig,
    TrainingConfig,
)
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.logging_utils import LoggingCounter, WandbTable, log_dataset_to_wandb
from robust_llm.models import WrappedModel
from robust_llm.models.model_utils import InferenceType
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks import BinaryCallback, CallbackInput, CallbackRegistry
from robust_llm.trainer import (
    AdversarialTrainer,
    AdversarialTrainerDatasetManagementCallback,
    AdversarialTrainerLoggingCallback,
    AdversarialTrainingStateCallback,
    RLLMTrainer,
)

CORE_CHECKPOINT_FILES = (
    CONFIG_NAME,
    OPTIMIZER_NAME,
    "rng_state.pth",
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)

ADV_STATE_NAME = "adversarial_training_state.json"


def nested_list_to_tuple(nested_list: list) -> tuple:
    return tuple(
        nested_list_to_tuple(sub_list) if isinstance(sub_list, list) else sub_list
        for sub_list in nested_list
    )


@dataclasses.dataclass
class Training:
    config: TrainingConfig
    train_rllm_dataset: RLLMDataset
    eval_rllm_dataset: dict[str, RLLMDataset]
    victim: WrappedModel
    model_name_to_save: str  # Used for saving the model to disk/hf
    environment_config: EnvironmentConfig
    evaluation_config: EvaluationConfig
    run_name: str
    trainer: Optional[RLLMTrainer] = None

    @property
    def report_to(self) -> None | str | list[str]:
        # report_to defaults to "all", which sets up several callbacks, including
        # a WandbCallback which increments the wandb internal step whenever
        # it logs. While this does not strictly break our logging setup,
        # it makes it harder to debug logging and makes the wandb plots with
        # wandb internal step on the horizontal axis less easily interpretable.
        # Thus, we set it to "none" here.
        return ["none"]

    def __post_init__(self):
        metrics = [evaluate.load("accuracy")]

        num_classes = self.train_rllm_dataset.num_classes
        if num_classes == 2:
            metrics.extend(
                [
                    evaluate.load("precision"),
                    evaluate.load("recall"),
                    evaluate.load("f1"),
                ]
            )

        self.metrics = evaluate.combine(metrics)

        # expose underlying hf datasets, prepared for training
        self.hf_train = self.train_rllm_dataset.for_hf_trainer()
        self.hf_eval = {
            key: ds.for_hf_trainer() for key, ds in self.eval_rllm_dataset.items()
        }
        self.training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            optim=self.config.optimizer,
            gradient_checkpointing=self.config.gradient_checkpointing,
            # Using non-reentrant checkpointing avoids a warning and
            # is recommended in the PyTorch docs:
            # https://pytorch.org/docs/stable/checkpoint.html
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            hub_model_id=f"AlignmentResearch/robust_llm_{self.model_name_to_save}",
            use_cpu=self.environment_config.device == "cpu",
            report_to=self.report_to,
        )

    @property
    def train_batch_size(self) -> int:
        return self.victim.train_minibatch_size

    @property
    def eval_batch_size(self) -> int:
        return self.victim.eval_minibatch_size

    def setup_trainer(self) -> RLLMTrainer:
        inference_type = self.train_rllm_dataset.inference_type
        if inference_type == InferenceType.CLASSIFICATION:
            # We use the right_tokenizer here because training does not involve
            # autoregressive generation.
            data_collator = transformers.DataCollatorWithPadding(
                self.victim.right_tokenizer
            )
        elif inference_type == InferenceType.GENERATION:
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=self.victim.right_tokenizer, mlm=False, return_tensors="pt"
            )
        else:
            raise ValueError(f"Unsupported inference type: {inference_type}")

        compute_metrics = (
            self.compute_metrics
            if inference_type == InferenceType.CLASSIFICATION
            else None
        )
        self.trainer = RLLMTrainer(
            model=self.victim.model,
            args=self.training_arguments,
            train_dataset=self.hf_train,
            eval_dataset=self.hf_eval,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        # Since we didn't pass an accelerator when constructing the WrappedModel,
        # we need to add it here. We do not need to 'prepare' the model with the
        # accelerator because the Trainer handles that.
        self.victim.accelerator = self.trainer.accelerator

        self.victim_training_logging_counter = LoggingCounter(
            _name="victim_training",
        )
        self.trainer.add_callback(CustomLoggingWandbCallback(self))

        return self.trainer

    @property
    def checkpoint_files(self) -> tuple[str, ...]:
        return CORE_CHECKPOINT_FILES + (
            (
                SAFE_WEIGHTS_NAME
                if self.training_arguments.save_safetensors
                else WEIGHTS_NAME
            ),
        )

    def get_last_checkpoint(self) -> str | None:
        """Get the directory path to the most recent completed checkpoint.

        We scan the output directory for the following conditions:
        - It is a directory.
        - It starts with "checkpoint".
        - It contains all the files in self.checkpoint_files.
        """
        if not os.path.isdir(self.output_dir):
            return None
        checkpoints = [
            f.path
            for f in os.scandir(self.output_dir)
            if f.is_dir()
            and f.name.startswith("checkpoint")
            and all([sub_f in os.listdir(f) for sub_f in self.checkpoint_files])
        ]
        if len(checkpoints) == 0:
            return None
        return checkpoints[-1]

    def run_trainer(self) -> None:
        trainer = self.trainer
        assert trainer is not None

        if trainer.is_world_process_zero() and self.config.log_full_datasets_to_wandb:
            self.log_datasets()

        checkpoint = self.get_last_checkpoint()
        if checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)

        self.maybe_save_model_to_path_or_hf(
            path_prefix_or_hf=self.config.model_save_path_prefix_or_hf
        )

    def compute_metrics(self, eval_preds: EvalPrediction) -> dict:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        computed_metrics = self.metrics.compute(
            predictions=predictions, references=labels
        )

        if computed_metrics is None:
            raise ValueError("computed_metrics is None, exiting...")

        return computed_metrics

    def log_datasets(self) -> None:
        """Save the training and eval datasets to wandb."""
        if self.trainer is None:
            raise ValueError(
                "self.trainer should have been assigned by now, exiting..."
            )
        if self.trainer.train_dataset is None:
            raise ValueError(
                "self.trainer.train_dataset should have been "
                "assigned by now, exiting..."
            )
        if self.trainer.eval_dataset is None:
            raise ValueError(
                "self.trainer.eval_dataset should have been "
                "assigned by now, exiting..."
            )

        validation_dataset = self.trainer.eval_dataset["validation"]
        assert isinstance(self.trainer.train_dataset, Dataset)
        assert isinstance(validation_dataset, Dataset)
        log_dataset_to_wandb(self.trainer.train_dataset, "train_dataset")
        log_dataset_to_wandb(validation_dataset, "validation_dataset")

    def maybe_save_model_to_path_or_hf(
        self, path_prefix_or_hf: Optional[str], adv_tr_round: Optional[int] = None
    ) -> None:
        assert self.trainer is not None

        # Make sure everything is in sync before saving.
        self.trainer.accelerator.wait_for_everyone()

        adv_tr_round_str = (
            f"adv-training-round-{adv_tr_round}" if adv_tr_round is not None else None
        )

        if path_prefix_or_hf is None:
            logger.info(
                "Not saving the model/tokenizer since no save path was specified"
            )

        elif path_prefix_or_hf == "hf":
            assert self.trainer.args.hub_model_id is not None
            # This is a hack to make sure we have the properly FSDP-wrapped
            # version of the model for saving. Without this, only the inner
            # layers are wrapped, not the whole model, which causes size
            # mismatches when saving/loading.
            self.victim.model = self.trainer.model  # type: ignore
            self.victim.push_to_hub(
                repo_id=self.trainer.args.hub_model_id,
                revision=adv_tr_round_str,
                retries=self.config.upload_retries,
                cooldown_seconds=self.config.upload_cooldown,
            )

            # Record the saving on wandb.
            hf_name = self.trainer.args.hub_model_id
            logger.info("Saving the model/tokenizer to HuggingFace as %s", hf_name)
            if wandb.run is not None:
                wandb.run.summary["saved_hf_name"] = hf_name

        else:
            adv_suffix = adv_tr_round_str or ""
            model_dir = self.model_name_to_save + adv_suffix
            output_dir = Path(path_prefix_or_hf) / "models" / model_dir
            if wandb.run is not None:
                wandb.run.summary["saved_dir"] = str(output_dir)
            logger.info("Saving the model/tokenizer to %s", output_dir)
            self.victim.save_local(output_dir=output_dir)

    @property
    def output_dir(self) -> str:
        return f"trainer/{self.run_name}_{self.model_name_to_save}"


class AdversarialTrainingState:
    """State for adversarial training, including the current round and RNG states.

    This is useful for saving and loading the state of the adversarial training in
    a way that can be resumed later.
    """

    def __init__(
        self,
        training_attack_rng: Optional[random.Random],
        validation_attack_rng: Optional[random.Random],
        current_round: int = 0,
    ) -> None:
        self.current_round = current_round
        self.training_attack_rng = training_attack_rng
        self.validation_attack_rng = validation_attack_rng

    @property
    def report_to(self) -> None | str | list[str]:
        return None  # This resolves to 'all' in the training args

    def to_dict(self) -> dict:
        return {
            "current_round": self.current_round,
            "training_attack_rng": (
                self.training_attack_rng.getstate()
                if self.training_attack_rng is not None
                else None
            ),
            "validation_attack_rng": (
                self.validation_attack_rng.getstate()
                if self.validation_attack_rng is not None
                else None
            ),
        }

    def save(self, checkpoint_dir: str) -> None:
        json.dump(
            obj=self.to_dict(),
            fp=open(os.path.join(checkpoint_dir, ADV_STATE_NAME), "w"),
        )

    def load(self, checkpoint_dir: str) -> None:
        output_path = os.path.join(checkpoint_dir, ADV_STATE_NAME)
        with open(output_path, "r") as f:
            state = json.load(f)
            self.current_round = state["current_round"]
            if self.training_attack_rng is not None:
                self.training_attack_rng.setstate(
                    nested_list_to_tuple(state["training_attack_rng"])
                )
            if self.validation_attack_rng is not None:
                self.validation_attack_rng.setstate(
                    nested_list_to_tuple(state["validation_attack_rng"])
                )


@dataclasses.dataclass(kw_only=True)
# TODO: make sure kw_only is not breaking anything.
# I put it there because of this:
# https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class AdversarialTraining(Training):
    """
    Creates an AdversarialTrainer wrapped for logging, etc.

    Parameters:
        num_adv_training_rounds:
            Number of rounds of adversarial training to do.
            One round of adversarial training involves first finding some number
            of adversarial examples, adding them to an "augmented train set",
            and training on that for some number of epochs.
            NOTE: The first round doesn't include adversarial examples.
        validation_attack_config:
            Config for the attack to use in validation.
    """

    config: TrainingConfig
    validation_attack_config: AttackConfig

    def __post_init__(self):
        super().__post_init__()

        assert type(self.eval_rllm_dataset) is dict
        assert "validation" in self.eval_rllm_dataset

        # Callback used for evaluating the victim on the validation set
        # and for finding examples the victim gets incorrect for adv training.
        callback_name = self.evaluation_config.final_success_binary_callback
        callback = CallbackRegistry.get_binary_callback(callback_name)
        self.victim_success_binary_callback = callback

        assert self.config.adversarial is not None
        self.adversarial_config = self.config.adversarial

        self.state = None

    @property
    def checkpoint_files(self) -> tuple[str, ...]:
        return (
            CORE_CHECKPOINT_FILES
            + (
                (
                    SAFE_WEIGHTS_NAME
                    if self.training_arguments.save_safetensors
                    else WEIGHTS_NAME
                ),
            )
            + (ADV_STATE_NAME,)
        )

    @property
    def num_adversarial_training_rounds(self) -> int:
        return self.adversarial_config.num_adversarial_training_rounds

    @property
    def training_attack_config(self) -> AttackConfig:
        return self.adversarial_config.training_attack

    @property
    def skip_first_training_round(self) -> bool:
        return self.adversarial_config.skip_first_training_round

    @property
    def adv_use_balanced_sampling(self) -> bool:
        return self.adversarial_config.use_balanced_sampling

    @property
    def num_examples_to_generate_each_round(self) -> int:
        return self.adversarial_config.num_examples_to_generate_each_round

    @property
    def num_examples_to_log_to_wandb_each_round(self) -> int:
        return self.adversarial_config.num_examples_to_log_to_wandb_each_round

    @property
    def only_add_successful_adversarial_examples(self) -> bool:
        return self.adversarial_config.only_add_successful_adversarial_examples

    @override
    def setup_trainer(self) -> AdversarialTrainer:
        self.trainer = AdversarialTrainer(
            use_balanced_sampling=self.adv_use_balanced_sampling,
            model=self.victim.model,
            args=self.training_arguments,
            train_dataset=self.hf_train,
            eval_dataset=self.hf_eval,
            # We use the right_tokenizer here because training does not involve
            # autoregressive generation.
            data_collator=transformers.DataCollatorWithPadding(
                self.victim.right_tokenizer
            ),
            compute_metrics=self.compute_metrics,
            tokenizer=self.victim.right_tokenizer,
        )
        # Since we didn't pass an accelerator when constructing the WrappedModel,
        # we need to add it here. We do not need to 'prepare' the model with the
        # accelerator because the Trainer handles that.
        self.victim.accelerator = self.trainer.accelerator

        self.victim_training_logging_counter = LoggingCounter(
            _name="victim_training",
        )
        self.trainer.add_callback(AdversarialTrainingStateCallback(self))
        self.trainer.add_callback(AdversarialTrainerLoggingCallback(self))
        self.trainer.add_callback(AdversarialTrainerDatasetManagementCallback(self))

        return self.trainer

    @override
    def run_trainer(self) -> None:
        adversarial_trainer = self.trainer
        assert isinstance(adversarial_trainer, AdversarialTrainer)

        # Prepare attacks
        training_attack = create_attack(
            attack_config=self.training_attack_config,
            logging_name="training_attack",
            victim=self.victim,
        )
        validation_attack = create_attack(
            attack_config=self.validation_attack_config,
            logging_name="validation_attack",
            victim=self.victim,
        )
        self.state = AdversarialTrainingState(
            training_attack_rng=getattr(training_attack, "rng", None),
            validation_attack_rng=getattr(validation_attack, "rng", None),
        )

        if adversarial_trainer.is_world_process_zero():
            # At present, we always log training information to wandb
            assert wandb.run is not None
            if self.config.log_full_datasets_to_wandb:
                self.log_datasets()

        checkpoint = self.get_last_checkpoint()
        if checkpoint is not None and training_attack.REQUIRES_TRAINING:
            logger.warning(
                "Resumption not yet supported for attacks that require training. "
            )
            checkpoint = None
        if checkpoint is not None:
            logger.debug(f"Loading adversarial state from {checkpoint}")
            self.state.load(checkpoint)
            logger.debug(f"Resuming from round {self.state.current_round}")

        # Run the adversarial training loop
        table = WandbTable("adversarial_eval/table")
        for round in range(self.num_adversarial_training_rounds):
            if round < self.state.current_round:
                continue
            logger.info("Adversarial training round %s started ", round)
            self._log_debug_info()

            self.state.current_round = round
            # Can be useful for x axis in plots
            wandb.log({"adversarial_training_round": round}, commit=False)

            # Train for "one round" (i.e., num_train_epochs)
            # on the (eventually, adversarial example-augmented) train set
            # Note that the first round is just normal training on the train set
            if round == 0 and self.skip_first_training_round:
                logger.info("Skipping first round of training...")
            else:
                logger.info("Victim started training in round %s", round)
                self._log_debug_info()

                adversarial_trainer.train(resume_from_checkpoint=checkpoint)
                checkpoint = None  # Only resume from checkpoint in the first round
                logger.info("Victim finished training in round %s ", round)
                self._log_debug_info()

            # Set the model to eval mode for the attacks. Model is set to train mode by
            # HF Trainer during training, otherwise we want it in eval mode.
            self.victim.eval()

            # Train the train/validation attacks if they need training.
            logger.info(
                "Adversary (training_attack) started training in round %s ", round
            )
            self._log_debug_info()
            _maybe_train_attack(
                attack=training_attack,
                dataset=self.train_rllm_dataset,
                train_or_validation="train",
                round=round,
            )
            logger.info(
                "Adversary (training_attack) finished training in round %s ", round
            )
            self._log_debug_info()

            logger.info(
                "Adversary (validation_attack) started training in round %s", round
            )
            self._log_debug_info()
            _maybe_train_attack(
                attack=validation_attack,
                dataset=self.eval_rllm_dataset["validation"],
                train_or_validation="validation",
                round=round,
            )
            logger.info(
                "Adversary (validation_attack) finished training in round %s ", round
            )

            # Perform adversarial evaluation every round
            victim_log_counter = self.victim_training_logging_counter
            do_adversarial_evaluation(
                victim=self.victim,
                dataset=self.eval_rllm_dataset["validation"],
                attack=validation_attack,
                final_success_binary_callback=self.victim_success_binary_callback,
                num_examples_to_log_detailed_info=self.evaluation_config.num_examples_to_log_detailed_info,  # noqa: E501
                adv_training_round=round,
                victim_training_step_count=victim_log_counter.step_count,
                victim_training_datapoint_count=victim_log_counter.datapoint_count,
                global_step_count=victim_log_counter.root.step_count,
                global_datapoint_count=victim_log_counter.root.datapoint_count,
                wandb_table=table,
            )

            # Now generate adversarial examples using the training attack; possibly
            # select only successful ones; and add them to the training set so that they
            # are used in the next round of training.
            if round < self.num_adversarial_training_rounds - 1:
                assert (
                    len(self.train_rllm_dataset.ds)
                    >= self.num_examples_to_generate_each_round
                )
                input_rllm_dataset = self.train_rllm_dataset.get_random_subset(
                    self.num_examples_to_generate_each_round
                )
                # NOTE: .get_attacked_dataset should relabel the examples
                attacked_dataset, _ = training_attack.get_attacked_dataset(
                    input_rllm_dataset
                )

                new_adv_examples = attacked_dataset.as_adversarial_examples()
                logger.info(
                    "Generated new adv examples for training, size (all): %s",
                    len(new_adv_examples),
                )
                logger.debug(
                    "The first few new adversarial examples are:\n%s",
                    new_adv_examples.ds["text"][:3],
                )

                # Select the ones to actually add to the training set.
                if self.only_add_successful_adversarial_examples:
                    selected_new_adv_examples = _get_only_data_with_incorrect_preds(
                        dataset=new_adv_examples,
                        victim=self.victim,
                        victim_success_binary_callback=self.victim_success_binary_callback,  # noqa: E501
                    )
                else:
                    selected_new_adv_examples = new_adv_examples

                # Log stats and a subset of examples to wandb.
                wandb.log(
                    {
                        "misc/num_selected_training_attack_data": len(
                            selected_new_adv_examples
                        ),
                    },
                    commit=False,
                )
                log_dataset_to_wandb(
                    dataset=new_adv_examples.ds,
                    dataset_name=f"data/all_new_adv_examples_r_{round}",
                    max_n_examples=self.num_examples_to_log_to_wandb_each_round,
                )
                log_dataset_to_wandb(
                    dataset=selected_new_adv_examples.ds,
                    dataset_name=f"data/selected_new_adv_examples_r_{round}",
                    max_n_examples=self.num_examples_to_log_to_wandb_each_round,
                )

                # Report new adversarial examples to the trainer so that they can be
                # later used for training.
                adversarial_trainer.add_new_adversarial_examples(
                    selected_new_adv_examples.for_hf_trainer()
                )

            logger.info("Adversarial training round %s finished", round)
            self._log_debug_info()

            self.maybe_save_model_to_path_or_hf(
                path_prefix_or_hf=self.config.model_save_path_prefix_or_hf,
                adv_tr_round=round,
            )
        table.save()

    def _log_debug_info(self):
        logger.debug(
            "Current logging counts: %s",
            self.victim_training_logging_counter._parent,
        )


def _maybe_train_attack(
    attack: Attack,
    dataset: RLLMDataset,
    train_or_validation: str,
    round: int,
) -> None:
    assert train_or_validation in ["train", "validation"]
    if attack.REQUIRES_TRAINING:
        train_this_round = False
        train_frequency = attack.attack_config.train_frequency
        if train_frequency is None and round == 0:
            train_this_round = True
        elif train_frequency is not None and round % train_frequency == 0:
            train_this_round = True

        if train_this_round:
            logger.info(
                "Training the %s attack on round %s", train_or_validation, round
            )
            attack.train(dataset=dataset)


@torch.no_grad()
def _get_only_data_with_incorrect_preds(
    dataset: RLLMDataset,
    victim: WrappedModel,
    victim_success_binary_callback: BinaryCallback,
) -> RLLMDataset:
    """Returns a dataset with only the examples that the model got wrong.

    Args:
        dataset: The dataset to filter.
        victim: The model to evaluate.
        victim_success_binary_callback: The callback to use for evaluating the model.
    """
    victim.eval()

    callback_input = CallbackInput(
        input_data=dataset.ds["text"],
        clf_label_data=dataset.ds["clf_label"],
        gen_target_data=dataset.ds["gen_target"],
    )
    victim_out = victim_success_binary_callback(
        victim,
        callback_input,
    )
    victim_successes = victim_out.successes
    # Reduce the dataset to only the examples the model got wrong.
    subset_indices = [i for i, success in enumerate(victim_successes) if not success]

    if len(subset_indices) == 0:
        warnings.warn("Got empty dataset after filtering; all preds were correct.")

    return dataset.get_subset(subset_indices)
