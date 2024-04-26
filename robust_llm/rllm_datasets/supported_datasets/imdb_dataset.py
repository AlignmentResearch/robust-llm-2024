from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class IMDBDataset(RLLMDataset):
    """IMDB dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        """IMDB has two labels (neg and pos)."""
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """IMDB consists of a single PERTURBABLE chunk."""
        return ModifiableChunkSpec(ChunkType.PERTURBABLE)
