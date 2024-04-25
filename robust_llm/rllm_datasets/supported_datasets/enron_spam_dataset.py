from typing_extensions import override

from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class EnronSpamDataset(RLLMDataset):
    """enron_spam dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        """EnronSpam has two labels (0: ham and 1: spam)."""
        return 2

    @property
    @override
    def modifiable_chunks_spec(self):
        """EnronSpam consists of a single modifiable chunk."""
        return (True,)
