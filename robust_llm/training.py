import dataclasses
import evaluate
import numpy as np


from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)


class PrintIncorrectClassificationsCallback(TrainerCallback):
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

    def run_trainer(self):
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
        )
        # Add a callback to print incorrect classifications (needs trainer so it can predict())
        trainer.add_callback(PrintIncorrectClassificationsCallback(trainer))

        # Perform an initial evaluation, then train
        trainer.evaluate(eval_dataset=self.eval_dataset)
        trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
