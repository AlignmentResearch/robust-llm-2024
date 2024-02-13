"""Common building blocks for pipelines."""

from typing import Optional, Tuple

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertLMHeadModel,
    GPTNeoXForCausalLM,
    GPTNeoXForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
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
from robust_llm.utils import LanguageModel


def _prepare_model(
    model_name_or_path: str,
    is_pythia: bool,
    checkpoint: Optional[int],
) -> PreTrainedModel:
    # TODO(GH#103): make it compatible with tasks where num_labels > 2.

    if is_pythia:
        revision = "main"
        # One of the original Pythia checkpoints.
        if model_name_or_path.startswith("EleutherAI/pythia"):
            assert checkpoint is not None
            revision = f"step{checkpoint}"

        model = GPTNeoXForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
            num_labels=2,
        )
        assert isinstance(model, PreTrainedModel)
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        )

    return model


def _prepare_tokenizer(
    model_name_or_path: str,
    is_pythia: bool,
    checkpoint: Optional[int],
) -> PreTrainedTokenizerBase:
    if is_pythia:
        revision = "main"
        # One of the original Pythia checkpoints.
        if model_name_or_path.startswith("EleutherAI/pythia"):
            assert checkpoint is not None
            revision = f"step{checkpoint}"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            model_max_length=512,  # TODO: check this number
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return tokenizer


def _prepare_decoder(
    decoder_name: str,
    is_pythia: bool,
    checkpoint: Optional[int],
) -> PreTrainedModel:
    # TODO(GH#103): make it compatible with tasks where num_labels > 2.
    if "bert" in decoder_name:
        decoder = BertLMHeadModel.from_pretrained(decoder_name, is_decoder=True)
        assert isinstance(decoder, PreTrainedModel)
    elif is_pythia:
        revision = "main"
        # One of the original Pythia checkpoints.
        if decoder_name.startswith("EleutherAI/pythia"):
            assert checkpoint is not None
            revision = f"step{checkpoint}"
        decoder = GPTNeoXForCausalLM.from_pretrained(
            decoder_name,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
        )
        assert isinstance(decoder, PreTrainedModel)
        decoder.config.pad_token_id = decoder.config.eos_token_id
    else:
        raise ValueError(f"Unknown model name {decoder_name}")

    return decoder


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
        model=model,
        tokenizer=tokenizer,
        language_generator_name=args.experiment.environment.language_generator,
        ground_truth_label_fn=robust_llm_datasets.ground_truth_label_fn,
    )
