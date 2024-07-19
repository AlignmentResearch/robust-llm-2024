from robust_llm.config.configs import DatasetConfig
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.rllm_datasets.supported_datasets import (
    EnronSpamDataset,
    HelpfulHarmlessDataset,
    IMDBDataset,
    PasswordMatchDataset,
    PureGenerationDataset,
    StrongREJECTDataset,
    WordLengthDataset,
)

SUPPORTED_DATASETS: dict[str, type[RLLMDataset]] = {
    "AlignmentResearch/IMDB": IMDBDataset,
    "AlignmentResearch/PasswordMatch": PasswordMatchDataset,
    "AlignmentResearch/WordLength": WordLengthDataset,
    "AlignmentResearch/EnronSpam": EnronSpamDataset,
    "AlignmentResearch/StrongREJECT": StrongREJECTDataset,
    "PureGeneration": PureGenerationDataset,
    "AlignmentResearch/Helpful": HelpfulHarmlessDataset,
    "AlignmentResearch/Harmless": HelpfulHarmlessDataset,
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
