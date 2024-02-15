"""Common building blocks for pipelines."""

from typing import Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.configs import OverallConfig
from robust_llm.dataset_management.dataset_management import (
    RobustLLMDatasets,
    generate_robust_llm_datasets,
)
from robust_llm.dataset_management.tomita import make_language_generator
from robust_llm.dataset_management.tomita.tomita import Tomita
from robust_llm.model_utils import _prepare_decoder, _prepare_model, _prepare_tokenizer
from robust_llm.utils import LanguageModel


def prepare_victim_models(
    args: OverallConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Optional[PreTrainedModel]]:
    """Returns the victim model, tokenizer, and optionally decoder used in defenses."""

    print("Preparing model and tokenizer (and maybe decoder)...")

    model_name_or_path = args.experiment.environment.model_name_or_path
    decoder_name = args.experiment.environment.decoder_name
    is_pythia = args.experiment.environment.is_pythia
    checkpoint = args.experiment.training.checkpoint

    # Note that the implication is only in one direction (i.e. we can have fine-tuned
    # Pythia-based models that do not have "pythia" in their name).
    if "pythia" in model_name_or_path or "pythia" in (decoder_name or ""):
        assert is_pythia

    model = _prepare_model(
        model_name_or_path=model_name_or_path,
        is_pythia=is_pythia,
        checkpoint=checkpoint,
    ).to(
        args.experiment.environment.device  # type: ignore
    )
    tokenizer = _prepare_tokenizer(
        model_name_or_path=model_name_or_path,
        is_pythia=is_pythia,
        checkpoint=checkpoint,
    )

    decoder = None
    if decoder_name is not None:
        decoder = _prepare_decoder(
            decoder_name=decoder_name, is_pythia=is_pythia, checkpoint=checkpoint
        ).to(
            args.experiment.environment.device  # type: ignore
        )
        decoder_tokenizer = _prepare_tokenizer(
            model_name_or_path=decoder_name, is_pythia=is_pythia, checkpoint=checkpoint
        )

        # Assert that tokenizers are the same. This check is of course not ideal.
        assert tokenizer.get_vocab() == decoder_tokenizer.get_vocab()

    return model, tokenizer, decoder


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
    tokenizer: PreTrainedTokenizerBase,
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
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
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
        victim_model=model,
        victim_tokenizer=tokenizer,
        language_generator_name=args.experiment.environment.language_generator,
        ground_truth_label_fn=robust_llm_datasets.ground_truth_label_fn,
    )
