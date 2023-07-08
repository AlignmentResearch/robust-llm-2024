from dataclasses import dataclass, replace
from typing import Optional
from functools import partial

from torch.distributed.elastic.multiprocessing.errors import record
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
import simple_parsing
import wandb

from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.training import Training

@dataclass
class HParams:
    hf_training_args: TrainingArguments
    model_name: str = "EleutherAI/pythia-70m-deduped"
    revision: Optional[str] = None
    string_max_len = 30
    train_size=1000
    val_size=200
    test_size=200


def tokenize_dataset(dataset, tokenizer, max_length):
    # Padding seems necessary in order to avoid an error
    # TODO prepend bos_token when using GPTNeoX so we aren't pushing the model off
    # distribution also the pad token probobly should be something else in that case
    # though maybe it doesn't matter
    tokenized_data = tokenizer(dataset["text"], padding="max_length", max_length=max_length, truncation=True)
    return Dataset.from_dict({"text": dataset["text"], "label": dataset["label"], **tokenized_data})


@record
def main():
    hparams: HParams = simple_parsing.parse(HParams)

    hparams.hf_training_args = replace(  # TODO it would be better to make this actual
        # defualts but there is a weird interaction between huggingface 
        # and simple-parsing
        hparams.hf_training_args,
        num_train_epochs=5,
        report_to=["wandb"],
        logging_steps=1000,
        dataloader_drop_last=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hparams.model_name, num_labels=2
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        hparams.model_name,
        revision=hparams.revision,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    tomita1 = Tomita1(hparams.string_max_len)
    train_set, val_set, test_set = tomita1.generate_dataset(
        train_size=hparams.train_size,
        val_size=hparams.val_size,
        test_size=hparams.test_size,
    )

    tokenize = partial(
        tokenize_dataset, 
        tokenizer=tokenizer,
        max_length=hparams.string_max_len*2 # TODO This is a huristic that will bite us
        # in the ass.
    )

    tokenized_train_dataset = tokenize(train_set)
    tokenized_val_dataset = tokenize(val_set)
    tokenized_test_dataset = tokenize(test_set)

    print("tokenized train dataset", tokenized_train_dataset)
    print("tokenized val dataset", tokenized_val_dataset)
    print("tokenized test dataset", tokenized_test_dataset)

    training = Training(
        hf_training_args=hparams.hf_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        model=model,
    )
    training.run_trainer()


if __name__ == "__main__":
    main()
