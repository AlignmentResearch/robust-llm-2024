"""Common building blocks for pipelines."""

from typing import Optional, Tuple

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPTNeoXForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.configs import OverallConfig
from robust_llm.dataset_management.dataset_management import (
    RobustLLMDatasets,
    generate_robust_llm_datasets,
)
from robust_llm.dataset_management.tomita import make_language_generator
from robust_llm.dataset_management.tomita.tomita import Tomita


def prepare_victim_model_and_tokenizer(
    args: OverallConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print("Preparing model and tokenizer...")

    model_name = args.experiment.environment.model_name

    # TODO(GH#103): make it compatible with tasks where num_labels > 2.
    if "bert" in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    elif "pythia" in model_name:
        checkpoint_step_number: int = args.experiment.training.checkpoint
        checkpoint_string: str = f"step{checkpoint_step_number}"
        pythia_version = model_name.split("/")[-1]
        untyped_model = GPTNeoXForSequenceClassification.from_pretrained(
            model_name,
            revision=checkpoint_string,
            cache_dir=f"./{pythia_version}/{checkpoint_string}",
            use_cache=False,  # otherwise returns last key/values attentions
            num_labels=2,
        )
        assert isinstance(untyped_model, PreTrainedModel)
        model = untyped_model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=checkpoint_string,
            cache_dir=f"./{pythia_version}/{checkpoint_string}",
            use_fast=True,
            model_max_length=512,  # TODO: check this number
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    else:
        raise ValueError(f"Unknown model name {model_name}")

    return model, tokenizer


def prepare_language_generator(args: OverallConfig) -> Optional[Tomita]:
    print("Preparing language generator...")

    if args.experiment.environment.dataset_type == "tomita":
        return make_language_generator(
            args.experiment.environment.language_generator,
            args.experiment.environment.max_length,
        )

    return None


def prepare_datasets(
    args: OverallConfig,
    tokenizer: PreTrainedTokenizer,
    language_generator: Optional[Tomita],
) -> RobustLLMDatasets:
    print("Preparing datasets...")

    return generate_robust_llm_datasets(
        dataset_type=args.experiment.environment.dataset_type,
        language_generator=language_generator,
        tokenizer=tokenizer,
        environment_config=args.experiment.environment,
        training_config=args.experiment.training,
        dataset_generation_style=args.experiment.environment.dataset_generation_style,
        seed=args.experiment.environment.seed,
    )


def prepare_attack(
    args: OverallConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    robust_llm_datasets: RobustLLMDatasets,
    training: bool,
) -> Attack:
    print("Preparing attack...")

    if training:
        attack_config = args.experiment.training.iterative.training_attack
    else:
        attack_config = args.experiment.evaluation.evaluation_attack

    return create_attack(
        attack_config=attack_config,
        modifiable_chunks_spec=robust_llm_datasets.modifiable_chunks_spec,
        dataset_type=args.experiment.environment.dataset_type,
        model=model,
        tokenizer=tokenizer,
        language_generator_name=args.experiment.environment.language_generator,
    )
