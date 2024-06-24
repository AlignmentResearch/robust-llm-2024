from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.config.configs import (
    DatasetConfig,
    EnvironmentConfig,
    EvaluationConfig,
    TrainingConfig,
)
from robust_llm.models import GPTNeoXModel
from robust_llm.models.model_utils import InferenceType
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
    wrapped_model = GPTNeoXModel(
        model,
        tokenizer,
        accelerator=None,
        inference_type=InferenceType("classification"),
        train_minibatch_size=2,
        eval_minibatch_size=2,
        generation_config=None,
        keep_generation_inputs=True,
    )

    train = load_rllm_dataset(dataset_cfg, split="train").tokenize(tokenizer)
    validation = load_rllm_dataset(dataset_cfg, split="validation").tokenize(tokenizer)

    config = TrainingConfig(
        model_save_path_prefix_or_hf="test-save-path",
    )

    Training(
        config=config,
        train_rllm_dataset=train,
        eval_rllm_dataset={"validation": validation},
        victim=wrapped_model,
        model_name_to_save="test_model",
        environment_config=EnvironmentConfig(),
        evaluation_config=EvaluationConfig(),
    )


# TODO test that training improves performance
