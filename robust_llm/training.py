import dataclasses
import evaluate
import numpy as np

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


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
            output_dir="test_trainer", evaluation_strategy="epoch",
            num_train_epochs=1, report_to=["wandb"], logging_steps=1,
        )
        trainer = Trainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    # TODO finish https://huggingface.co/docs/transformers/training
