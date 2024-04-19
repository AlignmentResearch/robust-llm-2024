from robust_llm.configs import DatasetConfig, ExperimentConfig, OverallConfig
from robust_llm.pipelines.utils import prepare_victim_models
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset


def test_prepare_victim_models_num_classes():
    TEST_CASES = [
        # dataset_type, expected_num_classes
        ("AlignmentResearch/PasswordMatch", 2),
        ("AlignmentResearch/IMDB", 2),
    ]

    for dataset_type, expected_num_classes in TEST_CASES:
        config = OverallConfig(
            experiment=ExperimentConfig(
                dataset=DatasetConfig(dataset_type=dataset_type, n_train=10),
            )
        )

        train_dataset = load_rllm_dataset(config.experiment.dataset, split="train")
        num_classes = train_dataset.num_classes
        victim_model, _, _ = prepare_victim_models(config, num_classes=num_classes)

        assert victim_model.config.num_labels == expected_num_classes
