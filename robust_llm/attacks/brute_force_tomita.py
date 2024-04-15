from typing import Any, Optional, Tuple

from datasets import Dataset
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.configs import AttackConfig, EnvironmentConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.dataset_management.tomita.tomita_dataset_generator import (
    load_adversarial_dataset,
)


class BruteForceTomitaAttack(Attack):
    """Brute force attack for Tomita datasets."""

    REQUIRES_INPUT_DATASET = False
    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: AttackConfig,
        environment_config: EnvironmentConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        dataset_type: str,
        language_generator_name: str,
    ) -> None:
        """Constructor for BruteForceTomitaAttack.

        Args:
            attack_config: config of the attack
            environment_config: config of the environment
            modifiable_chunks_spec: Specification for which chunks of the
                original text can be modified
            dataset_type: used dataset type
            language_generator_name: regular language used with tomita dataset
                (tomita1, tomita2, tomita4, or tomita7)
        """
        super().__init__(attack_config, environment_config, modifiable_chunks_spec)

        assert modifiable_chunks_spec == (True,)

        if dataset_type != "tomita":
            raise ValueError(
                "Brute force attack not supported in dataset type "
                f"{dataset_type}, exiting..."
            )

        self.language_generator_name = language_generator_name
        self.brute_force_length = attack_config.brute_force_tomita_attack_config.length

    @override
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset] = None,
        max_n_outputs: Optional[int] = None,
    ) -> Tuple[Dataset, dict[str, Any]]:
        brute_force_dataset = load_adversarial_dataset(
            self.language_generator_name, self.brute_force_length
        ).shuffle(seed=self.attack_config.seed)

        if max_n_outputs is not None:
            brute_force_dataset = brute_force_dataset.select(range(max_n_outputs))

        return brute_force_dataset, {}
