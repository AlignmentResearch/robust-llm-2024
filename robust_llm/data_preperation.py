from transformers import PreTrainedTokenizer
from datasets import load_dataset
from dataclasses import dataclass


@dataclass
class Data:
    dataset_name: str

    def prepare_tokenized_dataset(self, tokenizer: PreTrainedTokenizer, seed: int = 42):
        """Load a dataset and tokenize it with the given tokenizer.

        name_or_path: For a classification dataset with a test and train split.
        tokenizer: A tokenizer that can be used to tokenize the dataset.
        """

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        datasets = load_dataset(self.dataset_name)
        tokenized_datasets = datasets.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
        eval_dataset = tokenized_datasets["test"].shuffle(seed=seed)
        return train_dataset, eval_dataset
