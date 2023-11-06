import pytest
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.dataset_management.tomita import make_language_generator
from robust_llm.training import Training
from robust_llm.utils import tokenize_dataset

# 10 is long enough for all Tomita languages to have several
# true and false examples, but is otherwise arbitrary.
MAX_LANGUAGE_LENGTH = 10


def test_basic_constructor():
    language_generator = make_language_generator(
        language_name="Tomita1", max_length=MAX_LANGUAGE_LENGTH
    )
    train_dataset, eval_dataset, _test_dataset = language_generator.generate_dataset(
        train_size=10,
        val_size=10,
        test_size=10,
    )

    model_name = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenized_train_dataset = Dataset.from_dict(
        tokenize_dataset(train_dataset, tokenizer)
    )
    tokenized_eval_dataset = Dataset.from_dict(
        tokenize_dataset(eval_dataset, tokenizer)
    )

    Training(
        hparams={},
        train_dataset=tokenized_train_dataset,
        eval_dataset={"eval": tokenized_eval_dataset},
        model=model,
    )


# TODO test that training improves performance
