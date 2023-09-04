from datasets import concatenate_datasets, Dataset
from transformers import Trainer

from robust_llm.utils import tokenize_dataset


class AdversarialTrainer(Trainer):
    def __init__(self, **trainer_kwargs):
        super().__init__(**trainer_kwargs)

        self.adversarial_examples: dict = {"adversarial_string": [], "true_label": []}
        self.adversarial_examples_seen_so_far: int = 0

    # Overrides
    def get_train_dataloader(self):
        # Augment the train set with the new adversarial examples
        if self.adversarial_examples["adversarial_string"] != []:

            # We only want the *new* examples
            new_adversarial_examples = {
                "text": self.adversarial_examples["adversarial_string"][
                    self.adversarial_examples_seen_so_far :
                ],
                "label": self.adversarial_examples["true_label"][
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

            # Update the pointer to the end of the adversarial examples
            self.adversarial_examples_seen_so_far = len(
                self.adversarial_examples["adversarial_string"]
            )

            print(f"train_dataset is now of size {len(self.train_dataset)}")

        return super().get_train_dataloader()
