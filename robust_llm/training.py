import dataclasses
import evaluate
import numpy as np

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback


class PrintIncorrectClassificationsCallback(TrainerCallback):

    def on_evaluate(self, args, state, control, **kwargs):
        print("Printing incorrect eval classifications (or 20 of them if there are too many):")
        model = kwargs["model"]
        eval_dataloader = kwargs["eval_dataloader"]

        # Get the incorrect predictions
        incorrect_predictions = []
        batch = eval_dataloader.dataset  # we do this full batch for simplicity
        batch = {k: v.to(args.device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = np.argmax(logits, axis=-1)
        incorrect_predictions.append(predictions != batch["labels"])

        # Print up to 20 of them
        incorrect_predictions = np.concatenate(incorrect_predictions)
        if incorrect_predictions.sum() > 20:
            print("Too many incorrect predictions to print. Printing the first 20.")
            incorrect_predictions = incorrect_predictions[:20]
        for i, is_incorrect in enumerate(incorrect_predictions):
            if is_incorrect:
                print(f"Example {i}:")
                print(f"Input: {batch['input_ids'][i]}")
                print(f"Label: {batch['labels'][i]}")
                print(f"Prediction: {predictions[i]}")
                print()


@dataclasses.dataclass
class Training:
    hparams: dict
    train_dataset: Dataset
    eval_dataset: Dataset
    model: AutoModelForSequenceClassification

    def __post_init__(self):
        self.metric = evaluate.load("accuracy")

    def run_trainer(self):
        hf_training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=1,
            eval_steps=32,
            evaluation_strategy="steps",
            logging_steps=1,
            report_to=["wandb"],
        )
        trainer = Trainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[PrintIncorrectClassificationsCallback],
        )
        # Perform an initial evaluation, then train
        trainer.evaluate(eval_dataset=self.eval_dataset)
        trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    # TODO finish https://huggingface.co/docs/transformers/training
