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

    # TODO(GH#103): make it compatible with tasks where num_labels > 2.

    model_name_or_path = args.experiment.environment.model_name_or_path

    is_pythia = args.experiment.environment.is_pythia
    # Note that the implication is only in one direction (i.e. we can have fine-tuned
    # Pythia-based models that do not have "pythia" in their name).
    if "pythia" in model_name_or_path:
        assert is_pythia

    if is_pythia:
        revision = "main"  # default value for `revision` argument.
        # One of the original Pythia checkpoints.
        if model_name_or_path.startswith("EleutherAI/pythia"):
            checkpoint_step_number: int = args.experiment.training.checkpoint
            revision = f"step{checkpoint_step_number}"

        model = GPTNeoXForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
            num_labels=2,
        )
        assert isinstance(model, PreTrainedModel)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            model_max_length=512,  # TODO: check this number
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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
