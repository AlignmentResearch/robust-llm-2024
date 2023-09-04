import dataclasses
import evaluate
import numpy as np

from datasets import concatenate_datasets, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from robust_llm.adversarial_trainer import AdversarialTrainer
from robust_llm.utils import tokenize_dataset


class PrintIncorrectClassificationsCallback(TrainerCallback):
    # TODO: log this correctly to wandb
    def __init__(self, trainer: Trainer):
        super().__init__()
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        eval_dataloader = kwargs["eval_dataloader"]

        # Get the incorrect predictions
        outputs = self.trainer.predict(test_dataset=self.trainer.eval_dataset)

        logits = outputs.predictions
        predictions = np.argmax(logits, axis=-1)
        labels = outputs.label_ids

        prediction_outcomes = predictions == labels
        incorrect_indices = np.where(prediction_outcomes == False)[0].astype(int)

        incorrectly_predicted_texts = [
            self.trainer.eval_dataset["text"][incorrect_index]
            for incorrect_index in incorrect_indices
        ]
        incorrectly_predicted_true_labels = [
            self.trainer.eval_dataset["label"][incorrect_index]
            for incorrect_index in incorrect_indices
        ]
        incorrectly_predicted_predicted_labels = [
            predictions[incorrect_index] for incorrect_index in incorrect_indices
        ]

        if len(incorrectly_predicted_texts) == 0:
            print("\nAll eval texts predicted correctly.\n")
            return

        if len(incorrectly_predicted_texts) > 20:
            print(
                f"\nPrinting 20 (of the {len(incorrectly_predicted_texts)}) incorrect predictions:"
            )
        else:
            print(
                f"\nPrinting the {len(incorrectly_predicted_texts)} incorrect predictions:"
            )

        for i in range(min(20, len(incorrectly_predicted_texts))):
            print("Incorrectly predicted text:", incorrectly_predicted_texts[i])
            print("True label:", incorrectly_predicted_true_labels[i])
            print("Predicted label:", incorrectly_predicted_predicted_labels[i])
            print()
        print()


@dataclasses.dataclass
class Training:
    hparams: dict
    train_dataset: Dataset
    eval_dataset: Dataset
    model: AutoModelForSequenceClassification
    train_epochs: int = 3
    eval_steps: int = 150
    logging_steps: int = 150

    def __post_init__(self):
        self.metric = evaluate.load("accuracy")

    def setup_trainer(self):
        hf_training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=self.train_epochs,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            report_to=["wandb"],
        )
        trainer = Trainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        # Add a callback to print incorrect classifications (needs trainer so it can predict())
        trainer.add_callback(PrintIncorrectClassificationsCallback(trainer))

        return trainer

    def run_trainer(self):
        # Set up the trainer
        trainer = self.setup_trainer()

        # Perform an initial evaluation, then train
        trainer.evaluate(eval_dataset=self.eval_dataset)
        trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)


@dataclasses.dataclass(kw_only=True)
# TODO: make sure kw_only is not breaking anything.
# I put it there because of this:
# https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class AdversarialTraining(Training):
    tokenizer: AutoTokenizer
    num_adversarial_training_rounds: int
    brute_force_test = False

    # Overrides
    def setup_trainer(self):
        hf_training_args = TrainingArguments(
            output_dir="adversarial_trainer",
            num_train_epochs=self.train_epochs,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            report_to=["wandb"],
        )
        trainer = AdversarialTrainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        # Add a callback to print incorrect classifications (needs trainer so it can predict())
        trainer.add_callback(PrintIncorrectClassificationsCallback(trainer))

        return trainer

    # Overrides
    def run_trainer(self):
        # Set up the trainer
        adversarial_trainer = self.setup_trainer()

        # Run the adversarial training loop
        for _ in range(self.num_adversarial_training_rounds):
            # Train for "one round" (i.e., num_train_epochs) on the train set
            adversarial_trainer.train()

            # Find where the model makes mistakes on the eval set
            incorrect_predictions = self.get_incorrect_predictions(
                adversarial_trainer, self.eval_dataset
            )

            # Check if we have perfect accuracy now. If so, we're done.
            if len(incorrect_predictions["adversarial_string"]) == 0:
                print("Model got perfect accuracy, so stopping adversarial training.")
                break

            # Add the incorrect predictions to the adversarial dataset
            # If we already added that incorrect prediction, don't add it again
            for text_string, text_true_label in zip(
                incorrect_predictions["adversarial_string"],
                incorrect_predictions["true_label"],
            ):
                if (
                    text_string
                    not in adversarial_trainer.adversarial_examples[
                        "adversarial_string"
                    ]
                ):
                    adversarial_trainer.adversarial_examples[
                        "adversarial_string"
                    ].append(text_string)
                    adversarial_trainer.adversarial_examples["true_label"].append(
                        text_true_label
                    )

    def get_incorrect_predictions(self, trainer: Trainer, dataset: Dataset):
        # Get model outputs on the dataset provided
        eval_dataloader = trainer.get_eval_dataloader(dataset)

        output = trainer.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics,
            # otherwise we defer to self.args.prediction_loss_only
            prediction_loss_only=None,
            metric_key_prefix="adversarial-eval",
        )

        logits, labels, metrics, num_samples = output

        # Extract the incorrect predictions
        predictions = np.argmax(logits, axis=-1)

        prediction_outcomes = predictions == labels
        incorrect_indices = np.where(prediction_outcomes == False)[0].astype(int)

        # Return the incorrectly predicted examples, along with their true labels
        incorrect_predictions = {"adversarial_string": [], "true_label": []}
        for incorrect_index in incorrect_indices:
            incorrect_predictions["adversarial_string"].append(
                dataset["text"][incorrect_index]
            )
            incorrect_predictions["true_label"].append(
                dataset["label"][incorrect_index]
            )

        return incorrect_predictions
