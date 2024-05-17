"""Common building blocks for pipelines."""

from robust_llm import logger
from robust_llm.attacks.attack import Attack
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.config.configs import ExperimentConfig
from robust_llm.models import WrappedModel


def prepare_attack(
    args: ExperimentConfig,
    victim: WrappedModel,
    training: bool,
) -> Attack:
    logger.info("Preparing attack...")

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
        victim=victim,
    )
