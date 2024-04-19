import dataclasses
import os
import warnings
from typing import Callable, Optional

import evaluate
import numpy as np
import torch
import transformers
import wandb
import wandb.util
from datasets import Dataset
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.callbacks import CustomLoggingWandbCallback
from robust_llm.configs import AttackConfig, EnvironmentConfig, EvaluationConfig
from robust_llm.dataset_management.dataset_management import (
    ModifiableChunksSpec,
    get_num_classes,
)
from robust_llm.dataset_management.tomita.tomita import Tomita
from robust_llm.evaluation import (
    do_adversarial_evaluation,
    get_prediction_logits_and_labels_and_maybe_flag_values,
)
from robust_llm.logging_utils import LoggingCounter, log_dataset_to_wandb
from robust_llm.trainer import (
    AdversarialTrainer,
    AdversarialTrainerDatasetManagementCallback,
    AdversarialTrainerLoggingCallback,
    TrainerWithBatchSizeStoring,
)
from robust_llm.utils import tokenize_dataset


@dataclasses.dataclass
class Training:
    experiment_name: str
    run_name: str
    job_type: str
    train_dataset: Dataset
    eval_dataset: dict[str, Dataset]
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_name_to_save: str  # Used for saving the model to disk/hf
    environment_config: EnvironmentConfig
    evaluation_config: EvaluationConfig
    train_epochs: int = 3
    learning_rate: float = 5e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    optimizer: str = "adamw_torch"
    gradient_checkpointing: bool = False
    eval_steps: Optional[int | float] = None
    logging_steps: int | float = 500
    save_strategy: str = "steps"
    save_steps: int | float = 500
    trainer: Optional[TrainerWithBatchSizeStoring] = None
    log_full_datasets_to_wandb: bool = False
    ground_truth_label_fn: Optional[Callable[[str], int]] = None
    seed: int = 42

    def __post_init__(self):
        metrics = [evaluate.load("accuracy")]

        num_classes = get_num_classes(self.environment_config.dataset_type)
        if num_classes == 2:
            metrics.extend(
                [
                    evaluate.load("precision"),
                    evaluate.load("recall"),
                    evaluate.load("f1"),
                ]
            )

        self.metrics = evaluate.combine(metrics)

    def setup_trainer(self) -> TrainerWithBatchSizeStoring:
        hf_training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=self.train_epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            optim=self.optimizer,
            gradient_checkpointing=self.gradient_checkpointing,
            # Using non-reentrant checkpointing avoids a warning and
            # is recommended in the PyTorch docs:
            # https://pytorch.org/docs/stable/checkpoint.html
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            seed=self.seed,
            hub_model_id=f"AlignmentResearch/robust_llm_{self.model_name_to_save}",
            # This defaults to "all", which sets up several callbacks, including
            # a WandbCallback which increments the wandb internal step whenever
            # it logs. While this does not strictly break our logging setup,
            # it makes it harder to debug logging and makes the wandb plots with
            # wandb internal step on the horizontal axis less easily interpretable.
            # Thus, we set it to "none" here.
            report_to=["none"],
            use_cpu=self.environment_config.device == "cpu",
        )

        self.trainer = TrainerWithBatchSizeStoring(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,  # type: ignore
            eval_dataset=self.eval_dataset,  # type: ignore
            data_collator=transformers.DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics,
        )
        self.victim_training_logging_counter = LoggingCounter(
            _name="victim_training",
        )
        self.trainer.add_callback(CustomLoggingWandbCallback(self))

        return self.trainer

    def run_trainer(self) -> None:
        trainer = self.trainer
        assert trainer is not None

        if trainer.is_world_process_zero() and self.log_full_datasets_to_wandb:
            self.log_datasets()

        trainer.train()

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
                "self.trainer.train_dataset should have been assigned by now, exiting..."  # noqa: E501
            )
        if self.trainer.eval_dataset is None:
            raise ValueError(
                "self.trainer.eval_dataset should have been assigned by now, exiting..."  # noqa: E501
            )

        in_train = "label" in self.trainer.train_dataset.column_names  # type: ignore
        in_eval = "label" in self.trainer.eval_dataset["validation"].column_names  # type: ignore  # noqa: E501
        assert in_train == in_eval

        log_dataset_to_wandb(self.trainer.train_dataset, "train_dataset")  # type: ignore  # noqa: E501
        log_dataset_to_wandb(
            self.trainer.eval_dataset["validation"], "validation_dataset"  # type: ignore  # noqa: E501
        )

    def maybe_save_model_to_path_or_hf(self, path_prefix_or_hf: Optional[str]) -> None:
        assert self.trainer is not None

        # Make sure everything is in sync before saving.
        self.trainer.accelerator.wait_for_everyone()
        # In case of FSDP, we need to make sure we get correct state_dict to save.
        state_dict = self.trainer.accelerator.get_state_dict(self.trainer.model)

        if self.trainer.is_world_process_zero():
            assert wandb.run is not None
            if path_prefix_or_hf is None:
                print("Not saving the model/tokenizer since no save path was specified")

            elif path_prefix_or_hf == "hf":
                # Make sure the model is saved before pushing to HuggingFace;
                # without that, it does not work with accelerate. The model is saved
                # here to default local directory.
                self.trainer._save(state_dict=state_dict)
                hf_name = self.trainer.args.hub_model_id
                wandb.run.summary["saved_hf_name"] = hf_name  # type: ignore
                print(f"Saving the model/tokenizer to HuggingFace as {hf_name}")
                self.trainer.push_to_hub()
                # Even though above line should push both model and tokenizer, in
                # practice tokenizer sometimes doesn't get pushed, so we do it
                # explicitly here.
                assert self.trainer.hub_model_id is not None
                self.tokenizer.push_to_hub(self.trainer.hub_model_id)

            else:
                output_dir = os.path.join(
                    path_prefix_or_hf, "models", self.model_name_to_save
                )
                wandb.run.summary["saved_dir"] = output_dir  # type: ignore
                print(f"Saving the model/tokenizer to {output_dir}")
                self.trainer._save(output_dir=output_dir, state_dict=state_dict)
                self.tokenizer.save_pretrained(output_dir)


@dataclasses.dataclass(kw_only=True)
# TODO: make sure kw_only is not breaking anything.
# I put it there because of this:
# https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class AdversarialTraining(Training):
    """
    Creates an AdversarialTrainer wrapped for logging, etc.

    Parameters:
        num_iterative_training_rounds:
            One round of adversarial training involves first finding some number
            of adversarial examples, adding them to an "augmented train set",
            and training on that for some number of epochs.
        dataset_type:
            The type of dataset to use. Either "tomita" or "tensor_trust".
        language_generator:
            The language generator which should be created to generate
            datapoints for training and evaluation.
            Only relevant in the Tomita setting.
        training_attack_config:
            Config for the attack to use in adversarial training.
        validation_attack_config:
            Config for the attack to use in validation.
        modifiable_chunks_spec:
            Specification for which chunks of the original text can be modified.
        num_examples_to_generate_each_round (int): The number of adversarial examples to
            generate each round for training.
        num_examples_to_log_to_wandb_each_round (int): The number of adversarial
            examples to log to wandb each round.
        skip_first_training_round:
            Whether to skip the first round of training.
            Useful for doing "exclusively" adversarial training.
        only_add_successful_adversarial_examples:
            Whether to add only successful adversarial examples to training set;
            otherwise, add all trials, successful or not.
    """

    num_iterative_training_rounds: int
    dataset_type: str
    language_generator: Optional[Tomita]
    training_attack_config: AttackConfig
    validation_attack_config: AttackConfig
    modifiable_chunks_spec: ModifiableChunksSpec
    num_examples_to_generate_each_round: int
    num_examples_to_log_to_wandb_each_round: int
    skip_first_training_round: bool = False
    use_balanced_sampling: bool = False
    only_add_successful_adversarial_examples: bool = True

    def __post_init__(self):
        super().__post_init__()

        assert type(self.eval_dataset) is dict
        assert "validation" in self.eval_dataset

        # Standardize the language generator name
        if self.language_generator is None:
            self.language_generator_name: str = (
                "(no language generator, tensor trust setting)"
            )
        else:
            self.language_generator_name = self.language_generator.name

        self.current_iterative_training_round: int = 0

    @override
    def setup_trainer(self) -> AdversarialTrainer:
        hf_training_args = TrainingArguments(
            output_dir="adversarial_trainer",
            num_train_epochs=self.train_epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            optim=self.optimizer,
            gradient_checkpointing=self.gradient_checkpointing,
            # Using non-reentrant checkpointing avoids a warning and
            # is recommended in the PyTorch docs:
            # https://pytorch.org/docs/stable/checkpoint.html
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            seed=self.seed,
            hub_model_id=f"AlignmentResearch/robust_llm_{self.model_name_to_save}",
            use_cpu=self.environment_config.device == "cpu",
        )
        self.trainer = AdversarialTrainer(
            use_balanced_sampling=self.use_balanced_sampling,
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=transformers.DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        self.victim_training_logging_counter = LoggingCounter(
            _name="victim_training",
        )
        self.trainer.add_callback(CustomLoggingWandbCallback(self))
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
            environment_config=self.environment_config,
            modifiable_chunks_spec=self.modifiable_chunks_spec,
            logging_name="training_attack",
            dataset_type=self.dataset_type,
            victim_model=self.model,
            victim_tokenizer=self.tokenizer,
            accelerator=adversarial_trainer.accelerator,
            language_generator_name=self.language_generator_name,
            ground_truth_label_fn=self.ground_truth_label_fn,
        )
        validation_attack = create_attack(
            attack_config=self.validation_attack_config,
            environment_config=self.environment_config,
            modifiable_chunks_spec=self.modifiable_chunks_spec,
            logging_name="validation_attack",
            dataset_type=self.dataset_type,
            victim_model=self.model,
            victim_tokenizer=self.tokenizer,
            accelerator=adversarial_trainer.accelerator,
            language_generator_name=self.language_generator_name,
            ground_truth_label_fn=self.ground_truth_label_fn,
        )

        if adversarial_trainer.is_world_process_zero():
            # At present, we always log training information to wandb
            assert wandb.run is not None
            if self.log_full_datasets_to_wandb:
                self.log_datasets()

        # Run the adversarial training loop
        for round in range(self.num_iterative_training_rounds):
            print(
                f"Iterative training round {round} started "
                f"at logging counts: {self.victim_training_logging_counter._parent}"
            )
            self.current_iterative_training_round = round
            # Can be useful for x axis in plots
            wandb.log({"iterative_training_round": round}, commit=False)

            # Train for "one round" (i.e., num_train_epochs)
            # on the (eventually, adversarial example-augmented) train set
            # Note that the first round is just normal training on the train set
            if round == 0 and self.skip_first_training_round:
                print("Skipping first round of training...")
            else:
                print(
                    f"Victim started training in round {round} "
                    f"at logging counts: {self.victim_training_logging_counter._parent}"
                )
                adversarial_trainer.train()
                print(
                    f"Victim finished training in round {round} "
                    f"at logging counts: {self.victim_training_logging_counter._parent}"
                )

            # Train the train/validation attacks if they need training.
            print(
                f"Adversary (training_attack) started training in round {round} "
                f"at logging counts: {self.victim_training_logging_counter._parent}"
            )
            _maybe_train_attack(
                attack=training_attack,
                dataset=self.train_dataset,
                train_or_validation="train",
                round=round,
            )
            print(
                f"Adversary (training_attack) finished training in round {round} "
                f"at logging counts: {self.victim_training_logging_counter._parent}"
            )

            print(
                f"Adversary (validation_attack) started training in round {round} "
                f"at logging counts: {self.victim_training_logging_counter._parent}"
            )
            _maybe_train_attack(
                attack=validation_attack,
                dataset=self.eval_dataset["validation"],
                train_or_validation="validation",
                round=round,
            )
            print(
                f"Adversary (validation_attack) finished training in round {round} "
                f"at logging counts: {self.victim_training_logging_counter._parent}"
            )

            # Perform adversarial evaluation every round
            do_adversarial_evaluation(
                model=self.model,
                tokenizer=self.tokenizer,
                accelerator=adversarial_trainer.accelerator,
                dataset=self.eval_dataset["validation"],
                ground_truth_label_fn=self.ground_truth_label_fn,
                num_generated_examples=self.evaluation_config.num_generated_examples,
                attack=validation_attack,
                batch_size=self.evaluation_config.batch_size,
                num_examples_to_log_detailed_info=self.evaluation_config.num_examples_to_log_detailed_info,  # noqa: E501
            )

            # Now generate adversarial examples using the training attack; possibly
            # select only successful ones; and add them to the training set so that they
            # are used in the next round of training.
            if round < self.num_iterative_training_rounds - 1:
                assert (
                    len(self.train_dataset) >= self.num_examples_to_generate_each_round
                )
                input_examples = self.train_dataset.shuffle().select(
                    range(self.num_examples_to_generate_each_round)
                )
                generated_adv_examples, _ = training_attack.get_attacked_dataset(
                    input_examples
                )

                # Relabel if labeling function is provided. Critical e.g. for
                # tensor_trust dataset.
                if self.ground_truth_label_fn is not None:
                    generated_adv_examples = generated_adv_examples.map(
                        lambda x: {
                            "label": self.ground_truth_label_fn(x["text"])  # type: ignore  # noqa: E501
                        }
                    )

                new_adv_examples = Dataset.from_dict(
                    tokenize_dataset(generated_adv_examples, self.tokenizer)
                )
                print(
                    "Generated new adv examples for training, size (all):",
                    len(new_adv_examples["text"]),
                )
                print(
                    "The first few new adversarial examples are:",
                    new_adv_examples["text"][:3],
                )

                # Select the ones to actually add to the training set.
                if self.only_add_successful_adversarial_examples:
                    selected_new_adv_examples = (
                        _get_only_data_with_incorrect_predictions(
                            dataset=new_adv_examples,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            batch_size=self.eval_batch_size,
                        )
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
                    new_adv_examples,
                    f"data/all_new_adv_examples_r_{round}",
                    max_n_examples=self.num_examples_to_log_to_wandb_each_round,
                )
                log_dataset_to_wandb(
                    selected_new_adv_examples,
                    f"data/selected_new_adv_examples_r_{round}",
                    max_n_examples=self.num_examples_to_log_to_wandb_each_round,
                )

                # Report new adversarial examples to the trainer so that they can be
                # later used for training.
                for text, true_label in zip(
                    selected_new_adv_examples["text"],
                    selected_new_adv_examples["label"],
                ):
                    adversarial_trainer.new_examples["text"].append(text)
                    adversarial_trainer.new_examples["label"].append(true_label)

            print(
                f"Iterative training round {round} finished "
                f"at logging counts: {self.victim_training_logging_counter._parent}"
            )


def _maybe_train_attack(
    attack: Attack,
    dataset: Dataset,
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
            print(f"Training the {train_or_validation} attack on round {round}")
            attack.train(dataset=dataset)


@torch.no_grad()
def _get_only_data_with_incorrect_predictions(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> Dataset:
    """Returns a dataset with only the examples that the model got wrong."""
    model.eval()

    pred_logits, pred_labels, _ = (
        get_prediction_logits_and_labels_and_maybe_flag_values(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    )

    labels = np.array(dataset["label"])
    indices_where_wrong = np.where(np.array(pred_labels) != labels)[0]

    if len(indices_where_wrong) == 0:
        warnings.warn("Got empty dataset after filering; all predictions were correct.")

    return dataset.select(indices_where_wrong)
