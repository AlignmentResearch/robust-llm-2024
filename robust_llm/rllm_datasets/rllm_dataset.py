from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar, overload

import datasets
import numpy as np
from accelerate import Accelerator
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from robust_llm import logger
from robust_llm.config.configs import DatasetConfig
from robust_llm.dist_utils import broadcast_list_of_ints, is_main_process
from robust_llm.models.model_utils import InferenceType
from robust_llm.rllm_datasets.dataset_utils import (
    EXPECTED_COLUMNS,
    cast_column_to_feature,
    check_revision_is_supported,
    construct_text_and_chunked_text,
    maybe_get_version,
    strip_leading_whitespace,
    tokenize_dataset,
)
from robust_llm.rllm_datasets.modifiable_chunk_spec import ModifiableChunkSpec

# TypeVar for the return type of methods that return a new RLLMDataset.
D = TypeVar("D", bound="RLLMDataset")


class RLLMDataset(ABC):
    """A class representing a dataset for robust LLM experiments.

    Mostly a wrapper around huggingface datasets, with some additional metadata.

    Attributes:
        ds: The underlying huggingface dataset object.
            Currently, we assume that a dataset has at least the following columns:
            - text: The input text.
            - chunked_text: The input text, chunked into modifiable and
            unmodifiable parts, matching 'modifiable_chunk_spec'.
            - clf_label: The classification label.
        num_classes: The number of classes in the dataset.
        modifiable_chunk_spec: Datasets consist of sections that can and cannot
            be modified in various ways. For example, we might want to leave the
            instructions intact while allowing part of the example content to be
            perturbed and the rest overwritten. modifiable_chunk_spec is a
            tuple of ChunkType, an enum that specifies whether each chunk is
            IMMUTABLE, PERTURBABLE, or OVERWRITABLE.
        is_tokenized: Whether the dataset has been tokenized, i.e., whether it
            has 'input_ids' and 'attention_mask' columns.
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
        assert dataset_config.revision is not None
        check_revision_is_supported(
            dataset_config.dataset_type, dataset_config.revision
        )
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_type = dataset_config.dataset_type
        self.version = maybe_get_version(
            dataset_config.dataset_type, dataset_config.revision
        )
        self.inference_type = InferenceType(dataset_config.inference_type)
        self.classification_as_generation = dataset_config.classification_as_generation
        ds = self._load_dataset(
            cfg=dataset_config,
            revision=self.version,
            split=split,
            tokenizer=tokenizer,
        )
        # Make sure the dataset has the expected columns
        assert {"text", "chunked_text", "clf_label", "gen_target"}.issubset(
            set(ds.column_names)
        )
        # filter out rows where the text is empty
        # https://github.com/AlignmentResearch/robust-llm/issues/662
        # TODO: Figure out why there are empty rows in the dataset
        original_len = len(ds)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        if len(ds) < original_len:
            logger.debug(f"Filtered out {original_len - len(ds)} empty rows")
        self.ds = ds

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Specifies how many classes there are in the dataset"""
        raise NotImplementedError("num_classes must be implemented")

    @property
    @abstractmethod
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """Specifies which parts of the dataset can be modified, and how."""

    @property
    def is_tokenized(self) -> bool:
        if "input_ids" not in self.ds.column_names:
            return False
        assert "input_ids" in self.ds.column_names
        assert "attention_mask" in self.ds.column_names
        assert self.tokenizer is not None
        return True

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
        """Load the untokenized dataset from huggingface.

        This first loads the raw dataset and then post-processes it.
        """

        dataset = self._load_raw_dataset(cfg, split, revision)
        actual_columns = set(dataset.column_names)
        assert (
            EXPECTED_COLUMNS <= actual_columns
        ), f"Expected columns {EXPECTED_COLUMNS}, got {actual_columns}"
        extra_columns = list(actual_columns - EXPECTED_COLUMNS)
        if extra_columns:
            # To be cautious, let's remove all extra columns.
            logger.warning(
                f"Removing extra columns {extra_columns} from dataset "
                f"{cfg.dataset_type} revision {revision}"
            )
            dataset = dataset.remove_columns(list(actual_columns - EXPECTED_COLUMNS))
        dataset = self._post_process_dataset(dataset, cfg)
        return dataset

    def _post_process_dataset(self, ds: Dataset, cfg: DatasetConfig) -> Dataset:
        """Post-process the dataset after loading it.

        Currently this involves
        - constructing 'text', and 'chunked_text' columns
            out of the 'instructions', 'content', and 'answer_prompt' columns.
        - Optionally stripping leading whitespace from the 'gen_target' column.
        - Optionally overriding the 'gen_target' column with a specified string.
        """
        ds = construct_text_and_chunked_text(ds)

        if cfg.gen_target_override is not None:
            assert cfg.inference_type == "generation"
            assert not cfg.classification_as_generation
            ds = ds.map(lambda x: {"gen_target": cfg.gen_target_override})
        if cfg.proxy_gen_target_override is not None:
            assert cfg.inference_type == "generation"
            assert not cfg.classification_as_generation
            ds = ds.map(lambda x: {"proxy_gen_target": cfg.proxy_gen_target_override})
        if cfg.strip_leading_whitespace:
            assert cfg.inference_type == "generation"
            ds = strip_leading_whitespace(ds)

        return ds

    def _load_raw_dataset(
        self, cfg: DatasetConfig, split: str, revision: str
    ) -> Dataset:
        """Load the raw dataset from huggingface.

        This is used to load the dataset without post-processing the columns.
        """
        if split == "train":
            if cfg.n_train == 0:
                raise ValueError(
                    "Cannot load train split when DatasetConfig.n_train is 0"
                )
            return self._load_split(
                cfg, split, n_examples=cfg.n_train, revision=revision
            )

        elif split == "validation":
            if cfg.n_val == 0:
                raise ValueError(
                    "Cannot load validation split when DatasetConfig.n_val is 0"
                )
            return self._load_split(cfg, split, n_examples=cfg.n_val, revision=revision)
        else:
            raise ValueError(f"Unknown split {split}")

    def _load_split(
        self, cfg: DatasetConfig, split: str, n_examples: int, revision: str
    ) -> Dataset:
        """Load a split of the dataset with a given number of examples.

        Args:
            cfg: The DatasetConfig specifying the dataset to load.
            split: The split of the dataset to load.
            n_examples: The number of examples to load.
            revision: The revision of the dataset to load.

        Returns:
            The loaded dataset split.
        """
        ds = datasets.load_dataset(
            path=cfg.dataset_type,
            name=cfg.config_name,
            revision=revision,
            # We use slice splits to load a subset of the dataset.
            # https://huggingface.co/docs/datasets/en/loading#slice-splits
            split=f"{split}[:{n_examples}]",
            # By setting 'reuse_cache_if_exists' instead of
            # 'reuse_dataset_if_exists', we avoid `datasets` reusing cached
            # operations between processes, which was hiding a bug in
            # get_random_subset.
            download_mode="reuse_cache_if_exists",
        )
        assert isinstance(ds, Dataset)
        if len(ds) == 0:
            raise ValueError(
                f"Split {split} of dataset {cfg.dataset_type} has no examples."
                " Are you sure that this dataset is supposed to have this split?"
            )
        return ds

    def get_random_subset(
        self: D,
        n: int,
        seed: int | None = None,
        generator: np.random.Generator | None = None,
    ) -> D:
        """Return an RLLMDataset with a random subset of the original dataset."""
        assert (seed is None) != (
            generator is None
        ), "Exactly one of {seed, generator} must be provided"
        # When using multiple GPUs, we want to choose the same subset across
        # processes, so we use RNG from the main process.
        indices = []
        if is_main_process():
            if seed is not None:
                generator = np.random.default_rng(seed)
            assert generator is not None
            indices = generator.choice(len(self.ds), n, replace=False).tolist()
        # Create a temporary accelerator for broadcasting
        accelerator = Accelerator()
        indices = broadcast_list_of_ints(indices, accelerator)
        return self.get_subset(indices)

    def get_subset(self: D, indices: Iterable[Any]) -> D:
        """Return an RLLMDataset with a subset of the original dataset.

        Iterable[Any] typing is to match Dataset.select
        """
        new_ds = self.ds.select(indices)
        return self.with_new_ds(new_ds)

    def with_attacked_text(self: D, attacked_text: Sequence[str]) -> D:
        """Returns a new RLLMDataset with the attacked text and attacked labels."""
        if "attacked_text" in self.ds.column_names:
            raise ValueError("Dataset already has attacked_text column")

        if len(attacked_text) != len(self.ds):
            raise ValueError("attacked_text must have the same length as the dataset")

        new_ds = self.ds.add_column(
            "attacked_text",
            attacked_text,
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        # Add initial 'attacked_' columns
        new_ds = new_ds.add_column(
            "attacked_clf_label",
            self.ds["clf_label"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        new_ds = new_ds.add_column(
            "attacked_gen_target",
            self.ds["gen_target"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )

        # Add proxy columns.
        new_ds = new_ds.add_column(
            "attacked_proxy_clf_label",
            self.ds["proxy_clf_label"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )
        new_ds = new_ds.add_column(
            "attacked_proxy_gen_target",
            self.ds["proxy_gen_target"],
            new_fingerprint=None,  # type: ignore  # (bug in datasets?)
        )

        # Make 'attacked_clf_label' have the same Feature as 'clf_label'
        new_ds = cast_column_to_feature(
            ds=new_ds,
            column_name="attacked_clf_label",
            feature=self.ds.features["clf_label"],
        )
        attacked_ds = self.with_new_ds(new_ds)
        return attacked_ds

    @overload
    def clf_label_to_gen_target(self, clf_label: int) -> str: ...

    @overload
    def clf_label_to_gen_target(self, clf_label: list[int]) -> list[str]: ...

    def clf_label_to_gen_target(
        self,
        clf_label: int | list[int],
    ) -> str | list[str]:
        """Convert one or more clf_label to gen_target.

        Uses the int2str method of the clf_label feature to convert the label(s).
        """
        feature = self.ds.features["clf_label"]
        assert isinstance(feature, datasets.ClassLabel)
        gen_target = feature.int2str(clf_label)

        assert isinstance(gen_target, str) or isinstance(gen_target, list)
        return gen_target

    @overload
    def gen_target_to_clf_label(self, gen_target: str) -> int: ...

    @overload
    def gen_target_to_clf_label(self, gen_target: list[str]) -> list[int]: ...

    def gen_target_to_clf_label(
        self,
        gen_target: str | list[str],
    ) -> int | list[int]:
        """Convert one or more clf_label to gen_target.

        Uses the str2int method of the clf_label feature to convert the label(s).
        """
        feature = self.ds.features["clf_label"]
        assert isinstance(feature, datasets.ClassLabel)
        clf_label = feature.str2int(gen_target)

        assert isinstance(clf_label, int) or isinstance(clf_label, list)
        return clf_label

    def with_new_ds(self: D, new_ds: Dataset) -> D:
        """Make a new RLLMDataset with the given huggingface dataset."""
        # We make a shallow copy of the RLLMDataset and replace the dataset.
        # This should be fine because the ds is the only attribute that might be
        # mutated and we're replacing it with a reference to a new object, but
        # if other mutable attributes are added to RLLMDataset then this may
        # need to be changed.
        new_dataset = copy.copy(self)
        new_dataset.ds = new_ds
        return new_dataset

    def tokenize(self: D, tokenizer: PreTrainedTokenizerBase) -> D:
        """Return a tokenized version of the dataset"""
        if self.is_tokenized:
            raise ValueError("Dataset is already tokenized")
        tokenized_ds = tokenize_dataset(self.ds, tokenizer)
        # TODO (ian): find a cleaner way to make the new RLLMDataset
        new_dataset = self.with_new_ds(tokenized_ds)
        new_dataset.tokenizer = tokenizer
        return new_dataset

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
        assert self.tokenizer is not None
        if self.inference_type == InferenceType.CLASSIFICATION:
            ds_for_trainer = self.ds.rename_column("clf_label", "label")
            trainer_cols = ["text", "input_ids", "attention_mask", "label"]
            unused_cols = [
                c for c in ds_for_trainer.column_names if c not in trainer_cols
            ]
            ds_for_trainer = ds_for_trainer.remove_columns(unused_cols)
            return ds_for_trainer

        elif self.inference_type == InferenceType.GENERATION:
            # For generation, we tokenize the gen_target and stick it on the end
            # of the input_ids/attention_mask columns.
            ds_for_trainer = self.ds.remove_columns(["clf_label"])
            # TODO(ian): Make this work for chat models (pass in WrappedModel?)
            tokenized_gen_target = self.tokenizer(self.ds["gen_target"])
            all_target_ids = tokenized_gen_target["input_ids"]
            target_masks = tokenized_gen_target["attention_mask"]
            assert isinstance(all_target_ids, list)
            assert isinstance(target_masks, list)

            input_ids = [
                prompt_ids + target_ids
                for prompt_ids, target_ids in zip(self.ds["input_ids"], all_target_ids)
            ]

            attention_mask = [
                prompt_mask + target_mask
                for prompt_mask, target_mask in zip(
                    self.ds["attention_mask"],
                    target_masks,
                )
            ]

            # Remove and readd columns to update them
            ds_for_trainer = ds_for_trainer.remove_columns(
                ["input_ids", "attention_mask"]
            )
            ds_for_trainer = ds_for_trainer.add_column(
                "input_ids",
                input_ids,
                new_fingerprint=None,  # type: ignore  # (bug in datasets)
            )
            ds_for_trainer = ds_for_trainer.add_column(
                "attention_mask",
                attention_mask,
                new_fingerprint=None,  # type: ignore  # (bug in datasets)
            )

            trainer_cols = ["input_ids", "attention_mask"]
            unused_cols = [
                c for c in ds_for_trainer.column_names if c not in trainer_cols
            ]
            ds_for_trainer = ds_for_trainer.remove_columns(unused_cols)
            return ds_for_trainer

        else:
            raise ValueError(f"Unsupported inference type: {self.inference_type}")

    def as_adversarial_examples(self: D) -> D:
        """Formats attacked dataset as adversarial examples.

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

        columns_to_keep = ["attacked_text", "attacked_clf_label", "attacked_gen_target"]
        columns_to_drop = [c for c in self.ds.column_names if c not in columns_to_keep]

        untokenized_new_ds = self.ds.map(remove_columns=columns_to_drop)
        untokenized_new_ds = (
            untokenized_new_ds.rename_column("attacked_text", "text")
            .rename_column("attacked_clf_label", "clf_label")
            .rename_column("attacked_gen_target", "gen_target")
        )

        assert self.tokenizer is not None
        new_ds = tokenize_dataset(untokenized_new_ds, self.tokenizer)
        adversarial_dataset = self.with_new_ds(new_ds)

        return adversarial_dataset

    def __len__(self) -> int:
        return len(self.ds)
