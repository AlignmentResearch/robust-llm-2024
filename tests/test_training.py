from typing import cast

from accelerate import Accelerator
from transformers import AutoTokenizer, GPTNeoXPreTrainedModel

from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.config.configs import (
    AdversarialTrainingConfig,
    DatasetConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import GPTNeoXModel
from robust_llm.models.model_utils import InferenceType
from robust_llm.pipelines.training_pipeline import run_training_pipeline
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.scoring_callbacks import build_binary_scoring_callback
from robust_llm.training import _get_only_data_with_incorrect_preds
from robust_llm.utils import FakeClassifierWithPositiveList


def test_get_only_data_with_incorrect_preds():
    cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        n_train=10,
        n_val=10,
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train = load_rllm_dataset(cfg, split="train").tokenize(tokenizer)

    # assume all are marked positive
    positives = tokenizer.__call__(
        train.ds["text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    accelerator = Accelerator()
    model = FakeClassifierWithPositiveList(tokenizer=tokenizer, positives=positives)
    # We fake the type with 'cast' because we are using a FakeClassifierWithPositiveList
    victim = GPTNeoXModel(
        cast(GPTNeoXPreTrainedModel, model),
        tokenizer,
        accelerator=accelerator,
        inference_type=InferenceType("classification"),
        train_minibatch_size=2,
        eval_minibatch_size=2,
        generation_config=None,
        family="gpt_neox",
    )

    subset_indices = [
        i for i, d in enumerate(train.ds) if d["clf_label"] == 0  # type: ignore
    ]
    expected_filtered_dataset = train.get_subset(subset_indices)
    scoring_callback_config = CallbackConfig(
        callback_name="successes_from_text", callback_return_type="binary"
    )
    callback = build_binary_scoring_callback(scoring_callback_config)
    filtered_dataset = _get_only_data_with_incorrect_preds(
        dataset=train,
        victim=victim,
        victim_success_binary_callback=callback,
    )

    assert filtered_dataset.ds["text"] == expected_filtered_dataset.ds["text"]
    assert filtered_dataset.ds["clf_label"] == expected_filtered_dataset.ds["clf_label"]


def test_training_pipeline_doesnt_crash():
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(),
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
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            n_train=2,
            n_val=2,
        ),
        training=TrainingConfig(model_save_path_prefix_or_hf=None),
    )
    run_training_pipeline(config)


def test_adv_training_pipeline_doesnt_crash():
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(
            evaluation_attack=RandomTokenAttackConfig(n_its=2),
        ),
        model=ModelConfig(
            # We use a finetuned model so that the classification head isn't
            # randomly initialized. It's fine to use a model that is already
            # partially trained; we are just testing that the attack doesn't
            # crash and need non-zero pre-attack accuracy for that purpose.
            name_or_path="AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            train_minibatch_size=2,
            eval_minibatch_size=3,
            minibatch_multiplier=1,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            n_train=2,
            n_val=2,
        ),
        training=TrainingConfig(
            model_save_path_prefix_or_hf=None,
            adversarial=AdversarialTrainingConfig(
                num_examples_to_generate_each_round=2,
                num_adversarial_training_rounds=2,
                training_attack=RandomTokenAttackConfig(n_its=2),
            ),
        ),
    )
    run_training_pipeline(config)
