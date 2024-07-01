from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class PureGenerationDataset(RLLMDataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        split: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        assert split in ("train", "validation")
        assert InferenceType(dataset_config.inference_type) == InferenceType.GENERATION
        assert not dataset_config.classification_as_generation
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_type = dataset_config.dataset_type
        seed_data = (
            list(range(dataset_config.n_train))
            if split == "train"
            else list(
                range(
                    dataset_config.n_train,
                    dataset_config.n_train + dataset_config.n_val,
                )
            )
        )
        self.ds = Dataset.from_list(
            [
                {
                    "text": "Just riff for me:\n",  # placeholder for pre-attack
                    "chunked_text": ["Just riff for me:\n"],
                    "seed": seed,
                    "clf_label": 0,
                    "gen_target": "0",
                }
                for seed in seed_data
            ]
        )
        if self.tokenizer is not None:
            self.tokenize(self.tokenizer)

    @property
    @override
    def num_classes(self) -> int:
        return 1

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        return ModifiableChunkSpec(
            ChunkType.OVERWRITABLE,
        )
