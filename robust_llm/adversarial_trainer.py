import datasets
import torch
import torch.utils.data
import transformers

from typing_extensions import override
from datasets import concatenate_datasets, Dataset
from transformers import Trainer
from typing import Optional

from robust_llm.utils import tokenize_dataset


class AdversarialTrainer(Trainer):
    def __init__(self, **trainer_kwargs):
        super().__init__(**trainer_kwargs)

        self.adversarial_examples: dict = {"text": [], "label": []}
        self.num_rounds_completed: int = 0

    @override
    def get_train_dataloader(self):
        # This method is called at the start of each training loop, when my_trainer.train() is called
        # In turn, the train_dataloader it returns is called at the start of each training epoch
        # https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L812

        train_dataset_plus_adv_examples = self.get_augmented_training_set()

        print(
            f"This round's train set has {len(train_dataset_plus_adv_examples)} examples"
        )

        # From here to end, copied from Trainer.get_train_dataloader(), with some modifications
        if train_dataset_plus_adv_examples is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator
        if transformers.utils.is_datasets_available() and isinstance(
            train_dataset_plus_adv_examples, datasets.Dataset
        ):
            train_dataset_plus_adv_examples = self._remove_unused_columns(
                train_dataset_plus_adv_examples, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(
            train_dataset_plus_adv_examples, torch.utils.data.IterableDataset
        ):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = transformers.trainer_utils.seed_worker

        return self.accelerator.prepare(
            torch.utils.data.DataLoader(
                train_dataset_plus_adv_examples, **dataloader_params  # type: ignore
            )
        )

    def get_augmented_training_set(self) -> Dataset:
        # Augment the train set with the new adversarial examples
        if self.adversarial_examples["text"]:
            # Tokenize the new examples
            tokenized_adversarial_examples = self.get_tokenized_adversarial_dataset()

            train_dataset_plus_adv_examples = concatenate_datasets(
                [
                    self.train_dataset,  # type: ignore
                    tokenized_adversarial_examples,
                ]
            )

        else:
            train_dataset_plus_adv_examples = self.train_dataset

        return train_dataset_plus_adv_examples  # type: ignore

    def get_tokenized_adversarial_dataset(self) -> Optional[Dataset]:
        if self.adversarial_examples["text"]:
            # Tokenize the new examples
            tokenized_adversarial_examples = Dataset.from_dict(
                tokenize_dataset(self.adversarial_examples, self.tokenizer)
            )

            assert (
                self.train_dataset.features.type  # type: ignore
                == tokenized_adversarial_examples.features.type
            )

            return tokenized_adversarial_examples

        else:
            return None
