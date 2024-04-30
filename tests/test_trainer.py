from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.config.configs import DatasetConfig, EnvironmentConfig, EvaluationConfig
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.training import Training


def test_basic_constructor():
    dataset_cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        n_train=10,
        n_val=10,
    )

    model_name = "EleutherAI/pythia-14m"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train = load_rllm_dataset(dataset_cfg, split="train").tokenize(tokenizer)
    validation = load_rllm_dataset(dataset_cfg, split="validation").tokenize(tokenizer)

    Training(
        experiment_name="test-experiment",
        job_type="test-job_type",
        run_name="test-run",
        train_rllm_dataset=train,
        eval_rllm_dataset={"validation": validation},
        model=model,
        tokenizer=tokenizer,
        model_name_to_save="test_model",
        environment_config=EnvironmentConfig(),
        evaluation_config=EvaluationConfig(),
    )


# TODO test that training improves performance
