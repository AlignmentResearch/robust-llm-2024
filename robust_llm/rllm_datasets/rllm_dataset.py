from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

import datasets
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from robust_llm.configs import DatasetConfig
from robust_llm.rllm_datasets.dataset_utils import (
    get_largest_version,
    tokenize_dataset,
    valid_tag,
)


class RLLMDataset(ABC):
    """A class representing a dataset for robust LLM experiments.

    Mostly a wrapper around huggingface datasets, with some additional metadata.

    Attributes:
        ds: The underlying huggingface dataset object.
            Currently, we assume that a dataset has at least the following columns:
            - text: The input text.
            - chunked_text: The input text, chunked into modifiable and
            unmodifiable parts, matching 'modifiable_chunks_spec'.
            - clf_label: The classification label.
        num_classes: The number of classes in the dataset.
        modifiable_chunks_spec: Datasets consist of sections that can and cannot
            be modified. For example, we might want to leave the instructions intact
            while allowing the example-specific data to be modified. The
            modifiable_chunk_spec is a tuple of booleans, where True indicates that
            the corresponding chunk can be modified.
        ground_truth_label_fn: A function that maps an input to its ground truth
            label. This is ideal to have because sometimes adversarial attacks can
            change the true label. Not all datasets can easily define such a
            function though, in which case the labels are not changed and we assume
            the attack does not change the labels.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        split: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """Initialize an RLLMDataset.

        Args:
            dataset_config: Specifies most properties of the dataset.
            split: The split of the dataset to use (e.g., "train" or "validation").
            tokenizer: The tokenizer to use for the dataset. If None, the dataset will
                not be tokenized.
        """
        assert split in ("train", "validation")
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_type = dataset_config.dataset_type
        self.version = self._maybe_get_version(
            dataset_config.dataset_type, dataset_config.revision
        )
        ds = self._load_dataset(
            cfg=dataset_config,
            revision=self.version,
            split=split,
            tokenizer=tokenizer,
        )
        # Make sure the dataset has the expected columns
        assert {"text", "chunked_text", "clf_label"}.issubset(set(ds.column_names))
        self.ds = ds

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Specifies how many classes there are in the dataset"""
        raise NotImplementedError("num_classes must be implemented")

    @property
    @abstractmethod
    def modifiable_chunks_spec(self) -> tuple[bool, ...]:
        """Specifies which parts of the dataset can be modified."""
        pass

    @property
    def is_tokenized(self) -> bool:
        if "input_ids" not in self.ds.column_names:
            return False
        assert "input_ids" in self.ds.column_names
        assert "attention_mask" in self.ds.column_names
        assert self.tokenizer is not None
        return True

    def ground_truth_label_fn(self, text: str, label: int) -> int:
        """If there is a simple rule for how the ground truth
        label depends on the text, this function should return it.
        By default, just returns the existing label.
        """
        return label

    def _maybe_get_version(self, dataset_type: str, revision: str) -> str:
        """If a semver version was specified, use that. Otherwise if 'main' was
        specified, use the latest version.

        NOTE: We don't simply use 'main' because we want to record the version
        used and avoid race conditions that could arise from separately loading
        'main' and looking up the most recent version.
        """
        if revision == "main":
            version = get_largest_version(dataset_type)
            if version is None:
                raise ValueError("No versions found for dataset")
            return str(version)
        elif valid_tag(revision):
            return revision
        else:
            raise ValueError(
                f"Invalid revision: {revision}."
                " Should be 'main' or a valid semver version."
            )

    def _load_dataset(
        self,
        cfg: DatasetConfig,
        split: str,
        revision: str,
        tokenizer: PreTrainedTokenizerBase | None,
    ) -> Dataset:
        """Load the dataset and maybe tokenize it."""
        untokenized_ds = self._load_untokenized_dataset(cfg, split, revision)
        if tokenizer is None:
            return untokenized_ds
        else:
            tokenized_ds = tokenize_dataset(untokenized_ds, tokenizer)
            return tokenized_ds

    def _load_untokenized_dataset(
        self, cfg: DatasetConfig, split: str, revision: str
    ) -> Dataset:
        """Load the raw dataset from huggingface."""
        if cfg.inference_type == "generation":
            raise NotImplementedError("Generation datasets not yet supported")
        if split == "train":
            if cfg.n_train == 0:
                raise ValueError(
                    "Cannot load train split when DatasetConfig.n_train is 0"
                )
            return self._load_split(cfg, split, n_examples=cfg.n_train)

        elif split == "validation":
            if cfg.n_val == 0:
                raise ValueError(
                    "Cannot load validation split when DatasetConfig.n_val is 0"
                )
            return self._load_split(cfg, split, n_examples=cfg.n_val)
        else:
            raise ValueError(f"Unknown split {split}")

    def _load_split(self, cfg: DatasetConfig, split: str, n_examples: int) -> Dataset:
        """Load a split of the dataset with a given number of examples.

        We clear the ClassLabel feature from the `clf_label` column because
        it interferes with generating new examples in adversarial training.

        Args:
            cfg: The DatasetConfig specifying the dataset to load.
            split: The split of the dataset to load.
            n_examples: The number of examples to load.

        Returns:
            The loaded dataset split.
        """
        ds = datasets.load_dataset(
            path=cfg.dataset_type,
            name=cfg.config_name,
            revision=cfg.revision,
            # We use slice splits to load a subset of the dataset.
            # https://huggingface.co/docs/datasets/en/loading#slice-splits
            split=f"{split}[:{n_examples}]",
        )
        assert isinstance(ds, Dataset)
        ds = self._remove_and_readd_clf_label(ds)
        return ds

    def _remove_and_readd_clf_label(self, ds: Dataset) -> Dataset:
        """Remove and re-add the clf_label column to clear the ClassLabel feature.

        We have to do this because the ClassLabel feature prevents concatenating
        new examples generated in adversarial training with the original dataset
        because the datasets.Features don't match.

        TODO (GH#319): Find a cleaner way to do this, either here or when
        generating adv examples.
        """

        clf_labels = ds["clf_label"]
        ds = ds.remove_columns("clf_label")
        ds = ds.add_column(
            name="clf_label",
            column=clf_labels,
            new_fingerprint=None,  # type: ignore
        )
        return ds

    def get_random_subset(self, n: int, seed: int | None = None) -> RLLMDataset:
        """Return an RLLMDataset with a random subset of the original dataset."""

        new_ds = self.ds.shuffle(seed=seed).select(range(n))
        return self.with_new_ds(new_ds)

    def get_subset(self, indices: Iterable[Any]) -> RLLMDataset:
        """Return an RLLMDataset with a subset of the original dataset.

        Iterable[Any] typing is to match Dataset.select
        """
        new_ds = self.ds.select(indices)
        return self.with_new_ds(new_ds)

    def with_attacked_text(self, attacked_text: Sequence[str]) -> RLLMDataset:
        """Returns a new RLLMDataset with the attacked text and attacked labels."""
        if "attacked_text" in self.ds.column_names:
            raise ValueError("Dataset already has attacked_text column")

        new_ds = self.ds.add_column(
            "attacked_text",
            attacked_text,
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        # If the dataset allows for recomputation of the ground truth label, use it.
        attacked_labels = self.maybe_recompute_labels(
            new_ds["attacked_text"],
            new_ds["clf_label"],
        )
        new_ds = new_ds.add_column(
            "attacked_clf_label",
            attacked_labels,
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        attacked_ds = self.with_new_ds(new_ds)
        return attacked_ds

    def with_new_ds(self, new_ds: Dataset) -> RLLMDataset:
        """Make a new RLLMDataset with the given huggingface dataset."""
        # We make a shallow copy of the RLLMDataset and replace the dataset.
        # This should be fine because the ds is the only attribute that might be
        # mutated and we're replacing it with a reference to a new object, but
        # if other mutable attributes are added to RLLMDataset then this may
        # need to be changed.
        new_dataset = copy.copy(self)
        new_dataset.ds = new_ds
        return new_dataset

    def tokenize(self, tokenizer: PreTrainedTokenizerBase) -> RLLMDataset:
        """Return a tokenized version of the dataset"""
        if self.is_tokenized:
            raise ValueError("Dataset is already tokenized")
        tokenized_ds = tokenize_dataset(self.ds, tokenizer)
        # TODO (ian): find a cleaner way to make the new RLLMDataset
        new_dataset = self.with_new_ds(tokenized_ds)
        new_dataset.tokenizer = tokenizer
        return new_dataset

    def maybe_recompute_labels(
        self, texts: Sequence[str], labels: Sequence[int]
    ) -> list[int]:
        """Recompute labels using the ground truth label function.

        If ground_truth_label_fn is not overridden, this will return the original
        labels.
        """
        # preconditions
        assert len(texts) == len(labels)

        new_labels = [
            self.ground_truth_label_fn(text, label)
            for (text, label) in zip(texts, labels)
        ]
        # postconditions
        assert len(new_labels) == len(labels)
        return new_labels

    def for_hf_trainer(self) -> Dataset:
        """Returns a datasets.Dataset that is suitable for the hf Trainer.

        This involves:
        - Ensuring the dataset is tokenized
        - Renaming the `clf_label` column to `label`
        - Reducing to a minimal set of columns to avoid column mismatches
            when concatenating datasets (e.g. for adversarial training).

        Returns:
            A datasets.Dataset with columns 'text', 'input_ids', 'attention_mask',
            and 'label'.
        """
        assert self.is_tokenized
        ds_for_trainer = self.ds.rename_column("clf_label", "label")
        trainer_cols = ["text", "input_ids", "attention_mask", "label"]
        unused_cols = [c for c in ds_for_trainer.column_names if c not in trainer_cols]
        ds_for_trainer = ds_for_trainer.remove_columns(unused_cols)

        return ds_for_trainer

    def as_adversarial_examples(self) -> RLLMDataset:
        """Returns a version of an attacked dataset that is formatted as
            adversarial examples.

        This method is necessary because we need to take an attacked dataset
        (one with `attacked_text` and `attacked_clf_label` columns) and convert
        it into something that is compatible with the training data (i.e., with
        `text` and `clf_label` columns).

        To clarify:
        - Takes a tokenized RLLMDataset (self) that has `attacked_text` and
            `attacked_clf_label` columns.
        - Returns a tokenized RLLMDataset where `attacked_text` has been renamed
            to 'text' and `attacked_clf_label` has been renamed to `clf_label`.
        - NOTE: We don't need 'chunked_text' because we don't further attack
            these examples, we only use them for training.

        TODO (ian): Check that we really don't need chunked_text
        TODO (ian): make the process involve fewer weird method calls.
        """
        if "attacked_text" not in self.ds.column_names:
            raise ValueError("Dataset does not have attacked_text column")
        if not self.is_tokenized:
            raise ValueError("Dataset is not tokenized")
        new_text = self.ds["attacked_text"]
        new_labels = self.ds["attacked_clf_label"]
        untokenized_new_ds = Dataset.from_dict(
            {"text": new_text, "clf_label": new_labels}
        )

        assert self.tokenizer is not None
        new_ds = tokenize_dataset(untokenized_new_ds, self.tokenizer)
        adversarial_dataset = self.with_new_ds(new_ds)

        return adversarial_dataset

    def __len__(self) -> int:
        return len(self.ds)
