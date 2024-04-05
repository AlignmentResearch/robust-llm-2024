from robust_llm.configs import EnvironmentConfig, ExperimentConfig, OverallConfig
from robust_llm.pipelines.utils import prepare_victim_models


def test_prepare_victim_models_num_classes():
    TEST_CASES = [
        # dataset_type, expected_num_classes
        ("tensor_trust", 2),
        ("hf/imdb", 2),
        ("hf/mteb/tweet_sentiment_extraction", 3),
    ]

    for dataset_type, expected_num_classes in TEST_CASES:
        config = OverallConfig(
            experiment=ExperimentConfig(
                environment=EnvironmentConfig(dataset_type=dataset_type)
            )
        )

        victim_model, _, _ = prepare_victim_models(config)

        assert victim_model.config.num_labels == expected_num_classes
