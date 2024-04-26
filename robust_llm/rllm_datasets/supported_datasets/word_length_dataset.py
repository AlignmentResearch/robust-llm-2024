from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class WordLengthDataset(RLLMDataset):
    @property
    @override
    def num_classes(self) -> int:
        """WordLength has two classes: first word longer (0) or second (1)."""
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """
        WordLength has two chunks:
        1. The main chunk with instructions and the two words to compare (IMMUTABLE).
        2. The random tokens at the end of the example (OVERWRITABLE).
        """
        return ModifiableChunkSpec(ChunkType.IMMUTABLE, ChunkType.OVERWRITABLE)
