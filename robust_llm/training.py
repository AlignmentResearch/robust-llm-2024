import dataclasses
import glob
import math
import os
import shutil
from pathlib import Path
from typing import Optional

import evaluate
import numpy as np
import transformers
import wandb
import wandb.util
from accelerate import Accelerator
from datasets import Dataset
from transformers import EvalPrediction, TrainingArguments
from transformers.trainer import (
    CONFIG_NAME,
    OPTIMIZER_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from typing_extensions import override

from robust_llm import logger
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.callbacks import CustomLoggingWandbCallback
from robust_llm.config.configs import (
    AttackConfig,
    AttackScheduleConfig,
    EnvironmentConfig,
    EvaluationConfig,
    SaveTo,
    TrainingConfig,
)
from robust_llm.dist_utils import DistributedRNG
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.logging_utils import (
    LoggingCounter,
    WandbTable,
    log_dataset_to_wandb,
    wandb_log,
)
from robust_llm.models import WrappedModel
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import FlopCount
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks import build_binary_scoring_callback
from robust_llm.trainer import (
    ADV_FILES,
    AdversarialTrainer,
    AdversarialTrainerDatasetManagementCallback,
    AdversarialTrainingState,
    AdversarialTrainingStateCallback,
    EvaluationLoopCallback,
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

WEIGHT_FILES = (
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)


@dataclasses.dataclass
class Training:
    hash: str
    config: TrainingConfig
    train_rllm_dataset: RLLMDataset
    eval_rllm_dataset: dict[str, RLLMDataset]
    victim: WrappedModel
    model_name: str  # Used for saving the model to disk/hf
    environment_config: EnvironmentConfig
    evaluation_config: EvaluationConfig
    run_name: str
    trainer: Optional[RLLMTrainer] = None

    @property
    def report_to(self) -> str | list[str]:
        # report_to defaults to "all", which sets up several callbacks, including
        # a WandbCallback which increments the wandb internal step whenever
        # it logs. While this does not strictly break our logging setup,
        # it makes it harder to debug logging and makes the wandb plots with
        # wandb internal step on the horizontal axis less easily interpretable.
        # Thus, we set it to ["none"] here.
        return ["none"]

    def __post_init__(self):
        if self.config.save_name is None:
            self.model_name_to_save = self.model_name.replace("/", "_")
        else:
            self.model_name_to_save = self.config.save_name

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
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            group_by_length=self.config.group_by_length,
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

        logger.debug("Training arguments: %s", self.training_arguments)

    @property
    def per_device_train_batch_size(self) -> int:
        return min(
            len(self.train_rllm_dataset) // self.victim.num_processes,
            self.victim.train_minibatch_size,
        )

    @property
    def per_device_eval_batch_size(self) -> int:
        return min(
            len(self.eval_rllm_dataset) // self.victim.num_processes,
            self.victim.eval_minibatch_size,
        )

    @property
    def train_batch_size(self) -> int:
        return self.training_arguments.train_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.victim.gradient_accumulation_steps

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

    def get_last_checkpoint(self) -> str | None:
        """Get the directory path to the most recent completed checkpoint.

        We scan the output directory for the following conditions:
        - It is a directory.
        - It starts with "checkpoint".
        - It contains all the necessary files.
        """
        if not os.path.isdir(self.output_dir):
            return None
        checkpoints = [
            f.path
            for f in os.scandir(self.output_dir)
            if f.is_dir()
            and f.name.startswith("checkpoint")
            and all([sub_f in os.listdir(f) for sub_f in CORE_CHECKPOINT_FILES])
            and any([sub_f in os.listdir(f) for sub_f in WEIGHT_FILES])
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
        with self.victim.dont_count_flops():
            # We rely on HF to count FLOPs during training
            trainer.train(
                resume_from_checkpoint=(
                    checkpoint if self.environment_config.allow_checkpointing else False
                )
            )

        self.maybe_save_model_to_path_or_hf()

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

    def get_model_directory(self, revision: str = "") -> Path:
        """Get the directory to save the model to disk for future evaluation."""
        save_prefix = self.config.save_prefix
        model_dir = self.model_name_to_save + revision
        return Path(save_prefix) / "models" / model_dir

    def maybe_save_model_to_path_or_hf(
        self,
        adv_tr_round: Optional[int] = None,
    ) -> None:
        assert self.trainer is not None

        # Make sure everything is in sync before saving.
        self.trainer.accelerator.wait_for_everyone()

        save_to = self.config.save_to
        adv_tr_round_str = (
            f"-adv-training-round-{adv_tr_round}" if adv_tr_round is not None else ""
        )
        if save_to == SaveTo.NONE:
            logger.info(
                "Not saving the model/tokenizer since no save path was specified"
            )

        if save_to in (SaveTo.DISK, SaveTo.BOTH):
            output_dir = self.get_model_directory(adv_tr_round_str)
            if wandb.run is not None:
                wandb.run.summary["saved_dir"] = str(output_dir)
            logger.info("Saving the model/tokenizer to %s", output_dir)
            self.victim.save_local(output_dir=output_dir)

        if save_to in (SaveTo.HF, SaveTo.BOTH):
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

    @property
    def output_dir(self) -> str:
        return str(
            Path(self.config.save_prefix)
            / "trainer"
            / self.run_name
            / self.model_name_to_save
            / self.hash
        )


@dataclasses.dataclass
class AttackSchedule:
    config: AttackScheduleConfig
    num_rounds: int

    @property
    def attack_rounds(self) -> int:
        """The number of rounds for which we run the attack.

        We subtract two from the number of rounds because don't run the attack in
        round 0 (since it's just normal finetuning on clean examples) and in the
        last round (since there is no next round for which to generate examples).
        """
        return max(self.num_rounds - 2, 0)

    def __post_init__(self):
        """Process the attack schedule configuration.

        If num_round is 2 or below, we set the rate to 0 and ensure that
        start == end since we will run the attack at most once.
        """
        if self.config.end is not None and self.config.rate is not None:
            if self.attack_rounds == 0:
                if self.config.rate != 0:
                    raise ValueError("If num_rounds<=2, rate must be 0.")
                self.start = self.config.end
                self.end = self.config.end
                self.rate = 0.0
            else:
                self.start = self.config.end - int(
                    self.config.rate * self.attack_rounds
                )
                self.end = self.config.end
                self.rate = self.config.rate
        elif self.config.start is not None and self.config.rate is not None:
            if self.attack_rounds == 0:
                if self.config.rate != 0:
                    raise ValueError("If num_rounds<=2, rate must be 0.")
                self.start = self.config.start
                self.end = self.config.start
                self.rate = 0
            else:
                self.end = self.config.start + int(
                    self.config.rate * self.attack_rounds
                )
                self.start = self.config.start
                self.rate = self.config.rate
        elif self.config.start is not None and self.config.end is not None:
            if self.attack_rounds == 0:
                if self.config.start != self.config.end:
                    raise ValueError("If num_rounds<=2, start must equal end.")
                self.start = self.config.end
                self.end = self.config.end
                self.rate = 0
            else:
                self.rate = (self.config.end - self.config.start) / self.attack_rounds
                self.start = self.config.start
                self.end = self.config.end
        elif self.config.start is not None:
            self.start = self.config.start
            self.end = self.config.start
            self.rate = 0
        elif self.config.end is not None:
            self.start = self.config.end
            self.end = self.config.end
            self.rate = 0
        else:
            raise ValueError(f"Bad attack schedule config: {self.config}")

    def __getitem__(self, i: int) -> int:
        if i < 0 or i >= self.num_rounds - 1:
            raise IndexError(f"Index {i} out of bounds for {self.num_rounds} rounds")
        return self.start + int(i * self.rate)


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
        validation_iterations:
            Number of iterations to use in the validation attack.
    """

    config: TrainingConfig
    validation_attack_config: AttackConfig
    validation_iterations: int

    def __post_init__(self):
        super().__post_init__()

        assert type(self.eval_rllm_dataset) is dict
        assert "validation" in self.eval_rllm_dataset

        # Callback used for evaluating the victim on the validation set
        # and for finding examples the victim gets incorrect for adv training.
        callback_config = self.evaluation_config.final_success_binary_callback
        callback = build_binary_scoring_callback(callback_config)
        self.victim_success_binary_callback = callback

        assert self.config.adversarial is not None
        self.adversarial_config = self.config.adversarial

        self.rng = DistributedRNG(self.config.seed, self.victim.accelerator)
        self.round = 0
        self.total_flops = 0.0
        self.n_forward_calls = 0
        self.n_backward_calls = 0
        self.training_attack = None
        self.validation_attack = None
        self.attack_schedule = AttackSchedule(
            self.adversarial_config.attack_schedule,
            self.num_adversarial_training_rounds,
        )

        if self.training_arguments.gradient_accumulation_steps > 1:
            # When gradient_accumulation_steps > 1, Trainer may stop short
            # of the desired number of epochs due to a rounding issue:
            # https://github.com/huggingface/transformers/issues/33455
            # For ordinary training this isn't a big deal. For adversarial
            # training, AdversarialTrainer.maybe_update_adversarial_losses only
            # runs every full epoch, so we set max_steps (which overrides
            # num_train_epochs) to hit our desired epoch count.
            global_minibatch_size = (
                self.victim.num_processes * self.per_device_train_batch_size
            )
            num_minibatches_per_epoch = math.ceil(
                len(self.hf_train) / global_minibatch_size
            )
            num_minibatches = num_minibatches_per_epoch * self.config.num_train_epochs
            self.training_arguments.max_steps = math.ceil(
                num_minibatches / self.gradient_accumulation_steps
            )

    def get_total_training_steps(self) -> int:
        num_processes = Accelerator().num_processes
        assert self.config.adversarial is not None
        num_training_steps = 0
        n_train = min(
            len(self.train_rllm_dataset),
            self.config.adversarial.max_augmented_data_size,
        )
        for round in range(self.config.adversarial.num_adversarial_training_rounds):
            num_datapoints = n_train * self.config.num_train_epochs
            len_dataloader = math.ceil(
                num_datapoints
                / (self.training_arguments.train_batch_size * num_processes)
            )
            num_training_steps += max(
                len_dataloader // self.training_arguments.gradient_accumulation_steps,
                1,
            )
            n_train = min(
                n_train + self.config.adversarial.num_examples_to_generate_each_round,
                self.config.adversarial.max_augmented_data_size,
            )
        return num_training_steps

    def clean_checkpoints_and_return_valid(self) -> list[str]:
        """Clean up checkpoints and return the valid ones.

        We scan the output directory for the following conditions:
        - It is a directory.
        - It has the structure "round-{round}/checkpoint-{checkpoint}".
        - It contains all the necessary files.

        We delete incomplete and old checkpoints.
        """
        if not os.path.isdir(self.output_dir):
            return []
        possible_checkpoints = [
            f
            for f in glob.iglob(f"{self.output_dir}/round-*/checkpoint-*")
            if os.path.isdir(f)
        ]

        complete_checkpoints = [
            f
            for f in possible_checkpoints
            if all([sub_f in os.listdir(f) for sub_f in CORE_CHECKPOINT_FILES])
            and any([sub_f in os.listdir(f) for sub_f in WEIGHT_FILES])
            and all([sub_f in os.listdir(f) for sub_f in ADV_FILES])
        ]
        partial_checkpoints = list(
            set(possible_checkpoints) - set(complete_checkpoints)
        )

        if len(complete_checkpoints) == 0:
            return []
        # Sort by round number and then by checkpoint number
        complete_checkpoints = sorted(
            complete_checkpoints,
            key=lambda x: (
                int(x.split("/")[-2].split("-")[-1]),
                int(x.split("/")[-1].split("-")[-1]),
            ),
        )
        checkpoints_outside_limit = complete_checkpoints[
            : max(len(complete_checkpoints) - self.config.save_total_limit, 0)
        ]
        self.remove_checkpoints(partial_checkpoints + checkpoints_outside_limit)
        return complete_checkpoints

    def get_last_checkpoint(self) -> str | None:
        """Get the directory path to the most recent completed checkpoint."""
        complete_checkpoints = self.clean_checkpoints_and_return_valid()
        return complete_checkpoints[-1] if complete_checkpoints else None

    def remove_checkpoints(self, partial_checkpoints: list[str]) -> None:
        for partial_checkpoint in partial_checkpoints:
            assert (
                "round-" in partial_checkpoint and "checkpoint-" in partial_checkpoint
            )
            logger.warning(
                "Deleting checkpoint: %s",
                partial_checkpoint,
            )
            shutil.rmtree(partial_checkpoint)

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
    def loss_rank_weight(self) -> float:
        return self.adversarial_config.loss_rank_weight

    @property
    def max_adv_data_proportion(self) -> float:
        return self.adversarial_config.max_adv_data_proportion

    @property
    def max_augmented_data_size(self) -> int:
        return self.adversarial_config.max_augmented_data_size

    @property
    def sampling_decay(self) -> float:
        return self.adversarial_config.adv_sampling_decay

    @property
    def stopping_attack_success_rate(self) -> float:
        return self.adversarial_config.stopping_attack_success_rate

    @property
    def target_asr(self) -> Optional[float]:
        return self.adversarial_config.target_adversarial_success_rate

    @property
    def stopping_flops(self) -> float:
        return self.adversarial_config.stopping_flops

    @override
    def setup_trainer(self) -> AdversarialTrainer:
        self.trainer = AdversarialTrainer(
            use_balanced_sampling=self.adv_use_balanced_sampling,
            max_adv_data_proportion=self.max_adv_data_proportion,
            max_augmented_data_size=self.max_augmented_data_size,
            loss_rank_weight=self.loss_rank_weight,
            sampling_decay=self.sampling_decay,
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
            num_training_steps=self.get_total_training_steps(),
        )
        wandb_log({"num_training_steps": self.trainer.num_training_steps}, commit=False)
        # Since we didn't pass an accelerator when constructing the WrappedModel,
        # we need to add it here. We do not need to 'prepare' the model with the
        # accelerator because the Trainer handles that.
        self.victim.accelerator = self.trainer.accelerator

        self.victim_training_logging_counter = LoggingCounter(
            _name="victim_training",
        )
        self.trainer.add_callback(AdversarialTrainingStateCallback(self))
        self.trainer.add_callback(AdversarialTrainerDatasetManagementCallback(self))
        self.trainer.add_callback(EvaluationLoopCallback(self))

        return self.trainer

    @override
    def run_trainer(self) -> None:
        adversarial_trainer = self.trainer
        assert isinstance(adversarial_trainer, AdversarialTrainer)

        # Prepare attacks
        self.training_attack = create_attack(
            attack_config=self.training_attack_config,
            run_name=self.run_name + "_training_attack",
            logging_name="training_attack",
            victim=self.victim,
        )
        self.validation_attack = create_attack(
            attack_config=self.validation_attack_config,
            run_name=self.run_name + "_validation_attack",
            logging_name="validation_attack",
            victim=self.victim,
        )

        if adversarial_trainer.is_world_process_zero():
            # At present, we always log training information to wandb
            assert wandb.run is not None
            if self.config.log_full_datasets_to_wandb:
                self.log_datasets()

        checkpoint = self.get_last_checkpoint()
        if checkpoint is not None:
            logger.info(f"Loading adversarial state from {checkpoint}")
            state_to_resume = AdversarialTrainingState.load(
                checkpoint, self.victim.accelerator
            )
            starting_round = state_to_resume.current_round
            logger.info(f"Resuming from round {starting_round}")
        else:
            starting_round = 0
            state_to_resume = None

        # Run the adversarial training loop
        table = WandbTable("adversarial_eval/table")
        for self.round in range(starting_round, self.num_adversarial_training_rounds):
            logger.info("Adversarial training round %s started ", self.round)
            self._log_debug_info()

            adversarial_trainer.args.output_dir = (
                self.output_dir + f"/round-{self.round}"
            )
            logger.info(
                "Setting HF Trainer output dir (used for saving checkpoints) to "
                + adversarial_trainer.args.output_dir
            )
            # Can be useful for x axis in plots
            wandb_log({"adversarial_training_round": self.round}, commit=False)

            # Train for "one round" (i.e., num_train_epochs)
            # on the (eventually, adversarial example-augmented) train set
            # Note that the first round is just normal training on the train set
            if self.round == 0 and self.skip_first_training_round:
                logger.info("Skipping first round of training... (doing dummy step)")

                # HACK: To make sure the model is wrapped with FSDP properly, we run a
                # dummy train loop on a single datapoint. This works by setting the
                # loss to 0 for this step.
                train_dataset_temp = adversarial_trainer.train_dataset
                adversarial_trainer.train_dataset = train_dataset_temp.select(range(1))
                adversarial_trainer.do_dummy_train_step = True
                # If we're in round 0 and we're skipping the first round, we can't have
                # a checkpoint to resume from (since those are saved by .train)
                if checkpoint:
                    logger.info(
                        f"Skipping first round of training, so ignoring {checkpoint}"
                    )
                adversarial_trainer.train(resume_from_checkpoint=False)
                adversarial_trainer.train_dataset = train_dataset_temp
                adversarial_trainer.do_dummy_train_step = False

            else:
                logger.info("Victim started training in round %s", self.round)
                self._log_debug_info()
                # We resume from checkpoint only if:
                # 1. We allow checkpointing, and
                # 2. there is an existing checkpoint, and
                # 3. We haven't already resumed from a checkpoint in this
                #    experiment run.
                resume_from_checkpoint = (
                    checkpoint
                    if self.environment_config.allow_checkpointing
                    and (self.round == starting_round)
                    else False
                )
                if resume_from_checkpoint:
                    logger.debug(
                        f"Resuming from checkpoint {checkpoint} so "
                        "we don't need to update the augmented training set."
                    )
                else:
                    adversarial_trainer.update_augmented_training_set(
                        self.config.log_full_datasets_to_wandb, self.round
                    )
                if self.round == starting_round and (state_to_resume is not None):
                    logger.info(
                        "Applying loaded state to adversarial training for round %s",
                        self.round,
                    )
                    state_to_resume.apply_to_training(self)
                with self.victim.dont_count_flops():
                    # We rely on HF to count FLOPs during training
                    train_out = adversarial_trainer.train(
                        resume_from_checkpoint=resume_from_checkpoint
                    )

                logger.info("Victim finished training in round %s ", self.round)
                # Note that HF uses "flos" for FLOPs
                logger.debug(
                    "FLOPs for training in this round: %.2E",
                    train_out.metrics["total_flos"],
                )
                self.total_flops += train_out.metrics["total_flos"]
                self._log_debug_info()
                wandb_log(
                    {
                        "train/total_flops": self.total_flops,
                        "train/learning_rate": adversarial_trainer.get_learning_rates()[
                            0
                        ],
                    },
                    commit=False,
                )

            # HACK: Now we get the FSDP-wrapped model from the AdversarialTrainer
            self.victim.model = adversarial_trainer.model  # type: ignore
            # Set the model to eval mode for the attacks. Model is set to train mode by
            # HF Trainer during training, otherwise we want it in eval mode.
            self.victim.eval()

            # Perform adversarial evaluation every round
            victim_log_counter = self.victim_training_logging_counter
            compute_robustness_metric = self.evaluation_config.compute_robustness_metric
            round_metrics = do_adversarial_evaluation(
                victim=self.victim,
                dataset=self.eval_rllm_dataset["validation"],
                attack=self.validation_attack,
                n_its=self.validation_iterations,
                final_success_binary_callback=self.victim_success_binary_callback,
                num_examples_to_log_detailed_info=self.evaluation_config.num_examples_to_log_detailed_info,  # noqa: E501
                adv_training_round=self.round,
                victim_training_step_count=victim_log_counter.step_count,
                victim_training_datapoint_count=victim_log_counter.datapoint_count,
                global_step_count=victim_log_counter.root.step_count,
                global_datapoint_count=victim_log_counter.root.datapoint_count,
                wandb_table=table,
                # We don't use checkpointing of attacks during adversarial training
                resume_from_checkpoint=False,
                compute_robustness_metric=compute_robustness_metric,
            )

            if (
                round_metrics["adversarial_eval/attack_success_rate"]
                < self.stopping_attack_success_rate
            ):
                logger.info(
                    f"Stopping adversarial training at round {round} because attack "
                    f"success rate "
                    f"{round_metrics['adversarial_eval/attack_success_rate']} "
                    "is below the stopping threshold "
                    f"{self.stopping_attack_success_rate}"
                )
                break

            if self.total_flops > self.stopping_flops:
                logger.info(
                    f"Stopping adversarial training at round {round} because total"
                    f"FLOPs {self.total_flops:.2E} is above the stopping "
                    f"threshold {self.stopping_flops:.2E}"
                )
                break

            # Now generate adversarial examples using the training attack; possibly
            # select only successful ones; and add them to the training set so that they
            # are used in the next round of training.
            if self.round < self.num_adversarial_training_rounds - 1:
                n_its = self.attack_schedule[self.round]
                logger.info(
                    "The training attack strength for round %s is n_its=%s",
                    self.round,
                    n_its,
                )
                assert (
                    len(self.train_rllm_dataset.ds)
                    >= self.num_examples_to_generate_each_round
                )
                input_rllm_dataset = self.train_rllm_dataset.get_random_subset(
                    self.num_examples_to_generate_each_round,
                    accelerator=self.victim.accelerator,
                    generator=self.rng,
                )
                # NOTE: .get_attacked_dataset should relabel the examples
                with self.victim.flop_count_context() as flop_counter:
                    attack_out = self.training_attack.get_attacked_dataset(
                        input_rllm_dataset,
                        n_its=n_its,
                        resume_from_checkpoint=False,
                    )
                attacked_dataset = attack_out.dataset
                self.update_flops(flop_counter)

                new_adv_examples = attacked_dataset.as_adversarial_examples()
                logger.info(
                    "Generated new adv examples for training, size (all): %s",
                    len(new_adv_examples),
                )
                logger.debug(
                    "The first few new adversarial examples are:\n%s",
                    new_adv_examples.ds["text"][:3],
                )

                # Log stats and a subset of examples to wandb.
                wandb_log(
                    {
                        "misc/num_selected_training_attack_data": len(new_adv_examples),
                        "train/n_its": n_its,
                    },
                    commit=False,
                )
                log_dataset_to_wandb(
                    dataset=new_adv_examples.ds,
                    dataset_name=f"data/selected_new_adv_examples_r_{self.round}",
                    max_n_examples=self.num_examples_to_log_to_wandb_each_round,
                )

                # Report new adversarial examples to the trainer so that they can be
                # later used for training.
                adversarial_trainer.add_new_adversarial_examples(
                    new_adv_examples.for_hf_trainer()
                )

            logger.info("Adversarial training round %s finished", self.round)
            self._log_debug_info()

            self.maybe_save_model_to_path_or_hf(
                adv_tr_round=self.round,
            )

        table.save()

    def update_flops(self, flop_count: FlopCount):
        if (
            self.victim.accelerator is not None
            and not self.victim.accelerator.is_main_process
        ):
            return
        logger.debug("FLOPs for attacking in this round: %.2E", flop_count.flops)
        logger.debug("Forward calls: %s", flop_count.forward_calls)
        logger.debug("Backward calls: %s", flop_count.backward_calls)
        self.total_flops += flop_count.flops
        self.n_forward_calls += flop_count.forward_calls
        self.n_backward_calls += flop_count.backward_calls
        logger.debug(
            "Last 10 input shapes: %s",
            self.victim.input_shapes[-10:],
        )

    def _log_debug_info(self):
        logger.debug(
            "Current logging counts: %s",
            self.victim_training_logging_counter._parent,
        )
