from typing_extensions import override

from robust_llm.rllm_datasets.generation_scripts.password_match_generation import (
    RESPONSE_SEPARATOR,
)
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class PasswordMatchDataset(RLLMDataset):
    @property
    @override
    def num_classes(self) -> int:
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """
        PasswordMatch has three chunks:
        1. The context including instructions and the password (IMMUTABLE).
        2. The user's password (OVERWRITABLE).
        3. The closing response separator (default '\n---\n') (IMMUTABLE).
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.OVERWRITABLE,
            ChunkType.IMMUTABLE,
        )

    @override
    def ground_truth_label_fn(self, text: str, label: int) -> int:
        """The ground truth label is determined by comparing
        the user's password to the correct password.

        Args:
            text: The text of the example.
            label: The original label of the example (unused, included for
                compatibility).

        Returns:
            The ground truth label for the example.
        """
        # System password is string between first and second '"'
        # this is valid because it comes before the modifiable chunk
        system_password = text.split('"')[1]

        # User password is between first and last RESPONSE_SEPARATOR
        # this is valid because we have a separator before and after
        # the modifiable chunk.
        # TODO (ian): Work out how to decouple this from the specific
        # response separator used in generation
        start_response = text.index(RESPONSE_SEPARATOR)
        end_response = text.rindex(RESPONSE_SEPARATOR)
        user_password = text[start_response + len(RESPONSE_SEPARATOR) : end_response]

        return int(system_password == user_password)
