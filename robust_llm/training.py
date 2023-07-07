import dataclasses
import evaluate
import numpy as np

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


@dataclasses.dataclass
class Training:
    hf_training_args: TrainingArguments
    train_dataset: Dataset
    eval_dataset: Dataset
    model: AutoModelForSequenceClassification

    def __post_init__(self):
        self.metric = evaluate.load("accuracy")

    def run_trainer(self):
        trainer = Trainer(
            model=self.model,
            args=self.hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

