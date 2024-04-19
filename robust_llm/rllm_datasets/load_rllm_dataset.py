from robust_llm.configs import DatasetConfig
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.rllm_datasets.supported_datasets.imdb_dataset import IMDBDataset
from robust_llm.rllm_datasets.supported_datasets.password_match_dataset import (
    PasswordMatchDataset,
)
from robust_llm.rllm_datasets.supported_datasets.spam_dataset import SpamDataset
from robust_llm.rllm_datasets.supported_datasets.word_length_dataset import (
    WordLengthDataset,
)

SUPPORTED_DATASETS: dict[str, type[RLLMDataset]] = {
    "AlignmentResearch/IMDB": IMDBDataset,
    "AlignmentResearch/PasswordMatch": PasswordMatchDataset,
    "AlignmentResearch/WordLength": WordLengthDataset,
    "AlignmentResearch/Spam": SpamDataset,
}


def load_rllm_dataset(dataset_config: DatasetConfig, split: str) -> RLLMDataset:
    """Loads a dataset from huggingface based on its name."""
    dataset_type = dataset_config.dataset_type
    if dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {dataset_type} not supported.\n"
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )
    return SUPPORTED_DATASETS[dataset_type](dataset_config, split)
