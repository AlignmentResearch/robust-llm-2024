from typing_extensions import override
from datasets import concatenate_datasets, Dataset
from transformers import Trainer

from robust_llm.utils import tokenize_dataset


class AdversarialTrainer(Trainer):
    def __init__(self, **trainer_kwargs):
        super().__init__(**trainer_kwargs)

        self.adversarial_examples: dict = {"text": [], "label": []}
        self.adversarial_examples_seen_so_far: int = 0

    @override
    def get_train_dataloader(self):
        # This method is called at the start of each training loop, when my_trainer.train() is called
        # In turn, the train_dataloader it returns is called at the start of each training epoch
        # https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L812

        # Augment the train set with the new adversarial examples
        if self.adversarial_examples[
            "text"
        ] and self.adversarial_examples_seen_so_far < len(
            self.adversarial_examples["text"]
        ):
            # We only want the *new* examples
            new_adversarial_examples = {
                "text": self.adversarial_examples["text"][
                    self.adversarial_examples_seen_so_far :
                ],
                "label": self.adversarial_examples["label"][
                    self.adversarial_examples_seen_so_far :
                ],
            }

            # Tokenize the new examples
            tokenized_new_adversarial_examples = Dataset.from_dict(
                tokenize_dataset(new_adversarial_examples, self.tokenizer)
            )

            # Concatenate the new examples onto the train set
            assert (
                self.train_dataset.features.type
                == tokenized_new_adversarial_examples.features.type
            )

            self.train_dataset = concatenate_datasets(
                [
                    self.train_dataset,
                    tokenized_new_adversarial_examples,
                ]
            )

            # Update the count of the adversarial examples seen so far
            self.adversarial_examples_seen_so_far = len(
                self.adversarial_examples["text"]
            )

            print(f"train_dataset is now of size {len(self.train_dataset)}")

        return super().get_train_dataloader()
