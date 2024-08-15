from robust_llm.config.configs import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
)
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset


def test_victim_num_classes():
    TEST_CASES = [
        # dataset_type, expected_num_classes
        ("AlignmentResearch/PasswordMatch", 2),
        ("AlignmentResearch/IMDB", 2),
    ]

    for dataset_type, expected_num_classes in TEST_CASES:
        config = ExperimentConfig(
            experiment_type="evaluation",
            dataset=DatasetConfig(
                dataset_type=dataset_type, n_train=10, revision="2.1.0"
            ),
            model=ModelConfig(
                name_or_path="EleutherAI/pythia-14m",
                family="pythia",
                # We have to set this explicitly because we are not loading with Hydra,
                # so interpolation doesn't happen.
                inference_type="classification",
                train_minibatch_size=2,
                eval_minibatch_size=3,
                minibatch_multiplier=1,
            ),
            evaluation=EvaluationConfig(),
        )

        train_dataset = load_rllm_dataset(config.dataset, split="train")
        num_classes = train_dataset.num_classes
        victim = WrappedModel.from_config(
            config.model, accelerator=None, num_classes=num_classes
        )

        assert victim.config.num_labels == expected_num_classes
