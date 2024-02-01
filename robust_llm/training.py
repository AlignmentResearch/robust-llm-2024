import dataclasses
import os
from typing import Optional

import evaluate
import numpy as np
import wandb
import wandb.util
from datasets import Dataset
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from typing_extensions import override

from robust_llm.adversarial_trainer import (
    AdversarialTrainer,
    AdversarialTrainerDatasetManagementCallback,
    AdversarialTrainerLoggingCallback,
)
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.callbacks import CrossTrainRunStepRecordingWandbCallback
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.dataset_management.tomita.tomita import Tomita
from robust_llm.utils import (
    log_dataset_to_wandb,
    search_for_adversarial_examples,
    tokenize_dataset,
    yield_minibatch,
)


@dataclasses.dataclass
class Training:
    hparams: dict
    experiment_name: str
    run_name: str
    job_type: str
    train_dataset: Dataset
    eval_dataset: dict[str, Dataset]
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_name_to_save: str  # Used for saving the model to disk/hf
    train_epochs: int = 3
    eval_steps: int = 10
    logging_steps: int = 10
    trainer: Optional[Trainer] = None
    log_datasets_to_wandb: bool = False

    def __post_init__(self):
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")

        self.metrics = evaluate.combine([accuracy, precision, recall, f1])

    def setup_trainer(self) -> Trainer:
        hf_training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=self.train_epochs,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            hub_model_id=f"AlignmentResearch/robust_llm_{self.model_name_to_save}",
        )

        trainer = Trainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,  # type: ignore
            eval_dataset=self.eval_dataset,  # type: ignore
            compute_metrics=self.compute_metrics,
        )
        trainer.add_callback(CrossTrainRunStepRecordingWandbCallback)

        self.trainer = trainer

        return trainer

    def run_trainer(self) -> None:
        trainer = self.setup_trainer()

        if self.log_datasets_to_wandb:
            self.log_datasets()

        trainer.evaluate(eval_dataset=self.eval_dataset["validation"])  # type: ignore

        if self.train_epochs <= 0:
            print(f"Not training, since train_epochs={self.train_epochs}.")
            return

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
        to_log = {}

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

        if "label" in self.trainer.train_dataset.column_names:  # type: ignore
            train_table = wandb.Table(columns=["text", "label"])
            for text, label in zip(
                self.trainer.train_dataset["text"], self.trainer.train_dataset["label"]
            ):
                train_table.add_data(text, label)
            to_log["train_dataset"] = train_table
        else:
            train_table = wandb.Table(columns=["text"])
            for text in self.trainer.train_dataset["text"]:
                train_table.add_data(text)
            to_log["train_dataset"] = train_table

        label_in_validation = "label" in self.trainer.eval_dataset["validation"].column_names  # type: ignore  # noqa: E501
        if label_in_validation:
            eval_table = wandb.Table(columns=["text", "label"])
            for text, label in zip(
                self.trainer.eval_dataset["validation"]["text"],
                self.trainer.eval_dataset["validation"]["label"],
            ):
                eval_table.add_data(text, label)
            to_log["validation_dataset"] = eval_table
        else:
            eval_table = wandb.Table(columns=["text"])
            for text in self.trainer.eval_dataset["validation"]["text"]:
                eval_table.add_data(text)
            to_log["validation_dataset"] = eval_table

        wandb.log(to_log, commit=False)

    def maybe_save_model_to_path_or_hf(self, path_prefix_or_hf: Optional[str]) -> None:
        assert self.trainer is not None
        assert wandb.run is not None

        if path_prefix_or_hf is None:
            print("Not saving the model/tokenizer since no save path was specified")

        elif path_prefix_or_hf == "hf":
            hf_name = self.trainer.args.hub_model_id
            wandb.run.summary["saved_hf_name"] = hf_name
            print(f"Saving the model/tokenizer to HuggingFace as {hf_name}")
            self.trainer.push_to_hub()
            # Even though above line should push both model and tokenizer, in practice
            # tokenizer sometimes doesn't get pushed, so we do it explicitly here.
            assert self.trainer.hub_model_id is not None
            self.tokenizer.push_to_hub(self.trainer.hub_model_id)

        else:
            output_dir = os.path.join(
                path_prefix_or_hf, "models", self.model_name_to_save
            )
            wandb.run.summary["saved_dir"] = output_dir
            print(f"Saving the model/tokenizer to {output_dir}")
            self.trainer.save_model(output_dir)
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
        min_num_new_examples_to_add:
            When searching for adversarial examples in the brute force attack,
            we usually don't stop looking until we surpass this number.
            If we search the entire brute force dataset and don't find enough
            examples, we do stop looking.
            If we surpass max_num_search_for_adversarial_examples during the
            search, we also do stop looking.
        max_num_search_for_adversarial_examples:
            When searching for adversarial examples in the brute force dataset,
            we stop looking if we search more or equal to this number of examples.
            In practice we'll often search a few more than this
            number since we do the search in minibatches.
        adversarial_example_search_minibatch_size:
            The number of datapoints to consider at once when searching for
            adversarial examples.
        skip_first_training_round:
            Whether to skip the first round of training.
            Useful for doing "exclusively" adversarial training.
        use_probabilistic_robustness_check:
            Whether to determine model robustness by randomly selecting some
            examples from the brute force dataset and testing only on those,
            rather than the default of checking against the entire brute force
            dataset.
        only_add_successful_adversarial_examples:
            If true, then only add examples that the model got wrong. If false,
            then add random examples irrespective of whether the model got them
            right or wrong. Note that these random examples are still taken from the
            attack dataset, so unless the attack is very weak, the model is still
            likely to get a large proportion of these examples wrong.
    """

    num_iterative_training_rounds: int
    dataset_type: str
    language_generator: Optional[Tomita]
    training_attack_config: AttackConfig
    validation_attack_config: AttackConfig
    modifiable_chunks_spec: ModifiableChunksSpec
    min_num_new_examples_to_add: int
    max_num_search_for_adversarial_examples: int
    adversarial_example_search_minibatch_size: int
    skip_first_training_round: bool = False
    use_probabilistic_robustness_check: bool = False
    only_add_successful_adversarial_examples: bool = True

    def __post_init__(self):
        super().__post_init__()

        assert type(self.eval_dataset) is dict
        assert "validation" in self.eval_dataset

        # Standardize the language generator name
        if self.language_generator is None:
            self.language_generator_name = (
                "(no language generator, tensor trust setting)"
            )
        else:
            self.language_generator_name: str = self.language_generator.name

        self.current_iterative_training_round: int = 0

    @override
    def setup_trainer(self) -> AdversarialTrainer:
        hf_training_args = TrainingArguments(
            output_dir="adversarial_trainer",
            num_train_epochs=self.train_epochs,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            hub_model_id=f"AlignmentResearch/robust_llm_{self.model_name_to_save}",
        )
        trainer = AdversarialTrainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        trainer.add_callback(CrossTrainRunStepRecordingWandbCallback)
        trainer.add_callback(AdversarialTrainerLoggingCallback(self))
        trainer.add_callback(AdversarialTrainerDatasetManagementCallback(self))

        # Save the trainer as an attribute
        self.trainer = trainer

        return trainer

    @override
    def run_trainer(self) -> None:
        # Set up the trainer
        adversarial_trainer = self.setup_trainer()

        # Prepare attacks
        training_attack = create_attack(
            attack_config=self.training_attack_config,
            modifiable_chunks_spec=self.modifiable_chunks_spec,
            dataset_type=self.dataset_type,
            model=self.model,
            tokenizer=self.tokenizer,
            language_generator_name=self.language_generator_name,
        )
        validation_attack = create_attack(
            attack_config=self.validation_attack_config,
            modifiable_chunks_spec=self.modifiable_chunks_spec,
            dataset_type=self.dataset_type,
            model=self.model,
            tokenizer=self.tokenizer,
            language_generator_name=self.language_generator_name,
        )

        training_attack_dataset = Dataset.from_dict({})

        # Run the adversarial training loop
        for i in range(self.num_iterative_training_rounds):
            print("Starting training round", i)
            self.current_iterative_training_round = i

            # Train for "one round" (i.e., num_train_epochs) on the (eventually,
            # adversarial example-augmented) train set
            # Note that the first round is just normal training on the train set
            # NOTE: this is where wandb.init() is called by default
            if i == 0 and self.skip_first_training_round:
                print("Skipping first round of training...")
            else:
                if self.train_epochs == 0:
                    raise ValueError(
                        "Adversarial training should be done with >0 train epochs, exiting..."  # noqa: E501
                    )
                adversarial_trainer.train()

            if i == 0 or self.training_attack_config.repeat_attack_every_round:
                training_attack_dataset = Dataset.from_dict(
                    tokenize_dataset(
                        training_attack.get_attacked_dataset(self.train_dataset),
                        self.tokenizer,
                    )
                )

                self.training_attack_dataset = training_attack_dataset
                print(
                    "Training attack dataset has size",
                    len(training_attack_dataset["text"]),
                )
                print(
                    "The first few examples are:",
                    training_attack_dataset["text"][:5],
                )

            if i == 0 or self.validation_attack_config.repeat_attack_every_round:
                validation_attack_dataset = Dataset.from_dict(
                    tokenize_dataset(
                        validation_attack.get_attacked_dataset(
                            self.eval_dataset["validation"]
                        ),
                        self.tokenizer,
                    )
                )

                if not self.use_probabilistic_robustness_check:
                    # Save the attack dataset as one of the datasets to do eval on
                    self.eval_dataset["validation_attack_dataset"] = (
                        validation_attack_dataset
                    )

            if self.log_datasets_to_wandb:
                self.log_datasets()

            incorrect_predictions: dict[str, list[str]]
            number_examples_searched: int

            print("Searching for mistakes...")
            (
                incorrect_predictions,
                number_examples_searched,
            ) = search_for_adversarial_examples(
                adversarial_trainer,
                training_attack_dataset,
                min_num_new_examples_to_add=self.min_num_new_examples_to_add,
                max_num_search_for_adversarial_examples=self.max_num_search_for_adversarial_examples,  # noqa: E501
                adversarial_example_search_minibatch_size=self.adversarial_example_search_minibatch_size,  # noqa: E501
            )

            print(f"Model made {len(incorrect_predictions['text'])} mistakes.")
            print("Some examples are:", incorrect_predictions["text"][:5])

            examples_to_actually_add_to_train_set = incorrect_predictions

            if not self.only_add_successful_adversarial_examples:
                num_examples_to_add = (
                    self.min_num_new_examples_to_add
                    + self.adversarial_example_search_minibatch_size // 2
                )
                print(
                    f"Non-adversarial baseline: NOT adding those mistakes, instead adding the first {num_examples_to_add} random examples..."  # noqa: E501
                )
                examples_to_actually_add_to_train_set = next(
                    yield_minibatch(training_attack_dataset, num_examples_to_add)
                )

            wandb.log(
                {
                    "train/iterative_training_round": self.current_iterative_training_round,  # noqa: E501
                    "misc/number_examples_searched": number_examples_searched,
                    "misc/number_successful_attacks": len(
                        incorrect_predictions["text"]
                    ),
                },
                commit=False,
            )

            # Check if we have perfect accuracy now. If so, we're done.
            if len(incorrect_predictions["text"]) == 0:
                print(
                    f"~~~In round {i} of adversarial training, model got perfect accuracy on the {number_examples_searched} examples tried, so stopping adversarial training.~~~"  # noqa: E501
                )
                break

            # Log the successful attacks and the examples to add to the training set
            to_log = {}
            successful_attacks_table = wandb.Table(columns=["text", "correct label"])
            for text_string, correct_label in zip(
                incorrect_predictions["text"], incorrect_predictions["label"]
            ):
                successful_attacks_table.add_data(text_string, correct_label)
            to_log[f"successful_attacks_after_round_{i}"] = successful_attacks_table
            actual_examples_added_table = wandb.Table(columns=["text", "correct label"])
            for text_string, correct_label in zip(
                examples_to_actually_add_to_train_set["text"],
                examples_to_actually_add_to_train_set["label"],
            ):
                actual_examples_added_table.add_data(text_string, correct_label)
            to_log[f"examples_added_to_training_set_after_round_{i}"] = (
                actual_examples_added_table
            )
            wandb.log(to_log, commit=False)

            # Save the new examples to the adversarial trainer
            for text, true_label in zip(
                examples_to_actually_add_to_train_set["text"],
                examples_to_actually_add_to_train_set["label"],  # true label
            ):
                adversarial_trainer.new_examples["text"].append(text)
                adversarial_trainer.new_examples["label"].append(true_label)

            # Save the adversarial dataset as an eval set
            tokenized_new_examples = Dataset.from_dict(
                tokenize_dataset(adversarial_trainer.new_examples, self.tokenizer)
            )
            self.eval_dataset["all_examples_added_during_iterative_training"] = (
                tokenized_new_examples
            )

    @override
    def log_datasets(self) -> None:
        # First log the train and evaluation sets
        super().log_datasets()

        if self.use_probabilistic_robustness_check:
            return

        # Save the adversarial datasets to wandb tables
        if self.eval_dataset.get("validation_attack_dataset", None) is None:
            raise ValueError(
                "validation_attack_dataset should have been assigned by now, exiting..."  # noqa: E501
            )

        log_dataset_to_wandb(self.training_attack_dataset, "training_attack_dataset")
        log_dataset_to_wandb(
            self.eval_dataset["validation_attack_dataset"], "validation_attack_dataset"
        )
