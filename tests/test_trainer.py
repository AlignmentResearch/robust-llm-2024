import pytest
from datasets import Dataset
from robust_llm.language_generators import make_language_generator
from robust_llm.training import Training
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_basic_constructor():
    language_generator = make_language_generator(language_name="Tomita1", max_length=6)
    train_dataset, eval_dataset, _test_dataset = language_generator.generate_dataset(
        train_size=10, val_size=10, test_size=10,
    )

    model_name = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    Training(
        hparams={}, train_dataset=train_dataset, eval_dataset=train_dataset, model=model
    )
