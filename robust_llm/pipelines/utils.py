"""Common building blocks for pipelines."""

from typing import Optional, Tuple

from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.config.configs import ExperimentConfig
from robust_llm.model_utils import _prepare_decoder, _prepare_model, _prepare_tokenizer
from robust_llm.utils import LanguageModel


def prepare_victim_models(
    args: ExperimentConfig,
    num_classes: int,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Optional[PreTrainedModel]]:
    """Returns the victim model, tokenizer, and optionally decoder used in defenses."""

    print("Preparing model and tokenizer (and maybe decoder)...")

    model_name_or_path = args.model.name_or_path
    model_family = args.model.family
    revision = args.model.revision

    # TODO (ian): Load the decoder separately, rather than all in this one function
    try:
        decoder_name = args.defense.decoder.name_or_path  # type: ignore
        decoder_family = args.defense.decoder.family  # type: ignore
        decoder_revision = args.defense.decoder.revision  # type: ignore
    except AttributeError:
        decoder_name = None
        decoder_family = None
        decoder_revision = None

    model = _prepare_model(
        model_name_or_path=model_name_or_path,
        model_family=model_family,
        revision=revision,
        num_labels=num_classes,
    )
    tokenizer = _prepare_tokenizer(
        model_name_or_path=model_name_or_path,
        model_family=model_family,
        revision=revision,
    )

    decoder = None
    if decoder_name is not None:
        assert decoder_family is not None
        assert decoder_revision is not None
        decoder = _prepare_decoder(
            decoder_name=decoder_name,
            model_family=decoder_family,
            revision=decoder_revision,
        )
        decoder_tokenizer = _prepare_tokenizer(
            model_name_or_path=decoder_name,
            model_family=decoder_family,
            revision=decoder_revision,
        )

        # Assert that tokenizers are the same. This check is of course not ideal.
        assert tokenizer.get_vocab() == decoder_tokenizer.get_vocab()

    return model, tokenizer, decoder


def prepare_attack(
    args: ExperimentConfig,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    training: bool,
) -> Attack:
    print("Preparing attack...")

    if training:
        assert args.training is not None
        assert args.training.adversarial is not None
        logging_name = "training_attack"
        attack_config = args.training.adversarial.training_attack
    else:
        assert args.evaluation is not None
        logging_name = "eval_attack"
        attack_config = args.evaluation.evaluation_attack

    return create_attack(
        attack_config=attack_config,
        logging_name=logging_name,
        victim_model=model,
        victim_tokenizer=tokenizer,
        accelerator=accelerator,
    )
