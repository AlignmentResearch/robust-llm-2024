"""Common building blocks for pipelines."""

from typing import Optional, Tuple

from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.configs import OverallConfig
from robust_llm.model_utils import _prepare_decoder, _prepare_model, _prepare_tokenizer
from robust_llm.utils import LanguageModel


def prepare_victim_models(
    args: OverallConfig,
    num_classes: int,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Optional[PreTrainedModel]]:
    """Returns the victim model, tokenizer, and optionally decoder used in defenses."""

    print("Preparing model and tokenizer (and maybe decoder)...")

    model_name_or_path = args.experiment.environment.model_name_or_path
    model_family = args.experiment.environment.model_family
    revision = args.experiment.training.revision

    decoder_name = args.experiment.environment.decoder_name
    decoder_family = args.experiment.environment.decoder_family
    decoder_revision = args.experiment.environment.decoder_revision

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
    args: OverallConfig,
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    training: bool,
) -> Attack:
    print("Preparing attack...")

    if training:
        logging_name = "training_attack"
        attack_config = args.experiment.training.iterative.training_attack
    else:
        logging_name = "eval_attack"
        attack_config = args.experiment.evaluation.evaluation_attack

    return create_attack(
        attack_config=attack_config,
        logging_name=logging_name,
        victim_model=model,
        victim_tokenizer=tokenizer,
        accelerator=accelerator,
    )
