from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class StrongREJECTDataset(RLLMDataset):
    """StrongREJECT dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        """The StrongREJECT dataset doesn't support classification."""
        return 0

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """The StrongREJECT dataset consists of three chunks:

        1. The instructions, which are empty (IMMUTABLE)
        2. The forbidden_prompt (PERTURBABLE)
        3. The answer prompt (IMMUTABLE)
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.PERTURBABLE,
            ChunkType.IMMUTABLE,
        )
