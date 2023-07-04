import dataclasses
import evaluate
import numpy as np

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


@dataclasses.dataclass
class Training:
    hparams: dict

    metric = evaluate.load("accuracy")
    hf_training_args = TrainingArguments()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # TODO finish https://huggingface.co/docs/transformers/training
