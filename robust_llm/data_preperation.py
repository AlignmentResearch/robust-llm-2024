from transformers import PreTrainedTokenizer
from datasets import load_dataset
from dataclasses import dataclass


# 1 √
# 1*

# 2 √
# (10)*

# 3
# odd num of consecutive 1s always followed by even num of consecutive 0s

# 4 √
# any string not containing "000" as a substring

# 5
# even num of 0s and even num of 1s

# 6
# (num 1s) - (num 0s) is divisible by 3

# 7 √
# 0*1*0*1*

# LSTM can do all of them, transformer can only do 1, 2, 4, 7
# https://aclanthology.org/2020.emnlp-main.576.pdf


@dataclass
class Data:
    dataset_name: str

    def prepare_tokenized_dataset(
        self, tokenizer: PreTrainedTokenizer, seed: int = 42, mini: bool = False
    ):
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        datasets = load_dataset(self.dataset_name)
        tokenized_datasets = datasets.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
        eval_dataset = tokenized_datasets["test"].shuffle(seed=seed)

        if mini:
            train_dataset = train_dataset.select(range(1000))
            eval_dataset = eval_dataset.select(range(1000))

        return train_dataset, eval_dataset
