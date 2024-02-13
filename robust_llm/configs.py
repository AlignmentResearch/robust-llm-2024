from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

SHARED_DATA_DIR = "/robust_llm_data"


@dataclass
class BaselineTrainingConfig:
    """
    Configs used in baseline training.

    Attributes:
        proportion (float):
            The proportion of the brute force dataset to use for
            training, when running a baseline.
        non_iterative_baseline (bool): Whether to run a non-iterative baseline or not.
    """

    proportion: float = 0.1
    non_iterative_baseline: bool = False


@dataclass
class TextAttackAttackConfig:
    """
    Options specific for TextAttack attacks.

    Attributes:
        query_budget (int): Query budget per example.
        num_examples (int): Number of examples to attack. If -1, attack whole dataset.
        num_modifiable_words_per_chunk (Optional[int]): If set to an integer value, the
            attack will replace all content of each modifiable chunk with
            `num_modifiable_words_per_chunk` placeholder words which can be then
            modified by the attack. Otherwise, content is not modified at the start and
            the attack performs modifications on the original text.
        silent (bool): If silent, TextAttack will only print errors.
    """

    query_budget: int = 100
    num_examples: int = -1
    num_modifiable_words_per_chunk: Optional[int] = None
    silent: bool = True


@dataclass
class BruteForceTomitaAttackConfig:
    """
    Options specific for BruteForceTomita attacks.

    Attributes:
        length (int): Up to which length strings should be exhaustively tested.
    """

    length: int = 5


@dataclass
class RandomTokenAttackConfig:
    """Options specific for RandomToken attacks.

    Attributes:
        min_tokens (int): Minimum number of tokens to generate.
        max_tokens (int): Maximum number of tokens to generate.
    """

    min_tokens: int = 1
    max_tokens: int = 3


@dataclass
class AttackConfig:
    """
    Configs used in attack setup.

    Attributes:
        attack_type (str): The type of attack to use.
        repeat_attack_every_round (bool):
            Whether to repeat the attack every iterative training round or not.
        seed (int): Random seed for the attack.
        brute_force_tomita_attack_config (BruteForceTomitaAttackConfig):
            Config for BruteForceTomitaAttack.
        text_attack_attack_config (TextAttackAttackConfig):
            Config for TextAttackAttack.
        random_token_attack_attack_config (RandomTokenAttackConfig):
            Config for RandomTokenAttack.
    """

    attack_type: str = "identity"
    repeat_attack_every_round: bool = True
    seed: int = 0
    brute_force_tomita_attack_config: BruteForceTomitaAttackConfig = (
        BruteForceTomitaAttackConfig()
    )
    text_attack_attack_config: TextAttackAttackConfig = TextAttackAttackConfig()
    random_token_attack_attack_config: RandomTokenAttackConfig = (
        RandomTokenAttackConfig()
    )


@dataclass
class PerplexityDefenseConfig:
    """
    Configs used in perplexity-based defenses.

    Attributes:
        perplexity_threshold (Optional[float]):
            The perplexity threshold to use.
            If None, use the max perplexity in the train set.
        window_size (Optional[int]): Window size (if applicable).
        batch_size (int): Batch size to use for perplexity calculations.
        verbose (bool): Whether to print out the perplexity of each example.
    """

    perplexity_threshold: Optional[float] = None
    window_size: Optional[int] = None
    batch_size: int = 4
    verbose: bool = False

    @property
    def windowed(self) -> bool:
        return self.window_size is not None


@dataclass
class RetokenizationDefenseConfig:
    """
    Configs used in re-tokenization-based defenses.

    Attributes:
        drop_percentage (float): Percentage of the byte pair merges to drop.
        verbose (bool): Whether to print out the tokens of each example.
    """

    drop_percentage: float = 0.2
    verbose: bool = False


@dataclass
class ParaphraseDefenseConfig:
    """
    Configs used in paraphrase-based defenses.

    Attributes:
        model_name (str): Model to use for paraphrasing.
        meta_prompt (str): Meta-prompt to use for paraphrasing.
        temperature (float): Temperature to use for generation.
        verbose (bool): Verbosity.
        device (str): Device to store paraphrase model on.
        padding_side (str):
            Padding side when paraphrasing. Should be left for decoder models.
    """

    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    meta_prompt: str = "Paraphrase the following sentences: "
    temperature: float = 0.7
    verbose: bool = False
    device: str = "cuda:0"
    padding_side: str = "left"


@dataclass
class DefenseConfig:
    """
    Configs used in defense setup.

    Attributes:
        defense_type (str): The type of defense.
        seed (int): Random seed for the defense.
        perplexity_defense_config (PerplexityDefenseConfig):
            Configs for perplexity-based defenses.
        retokenization_defense_config (RetokenizationDefenseConfig):
            Configs for re-tokenization-based defenses.
        paraphrase_defense_config (ParaphraseDefenseConfig):
            Configs for paraphrase-based defenses.
    """

    defense_type: str = "identity"
    seed: int = 0
    perplexity_defense_config: PerplexityDefenseConfig = PerplexityDefenseConfig()
    retokenization_defense_config: RetokenizationDefenseConfig = (
        RetokenizationDefenseConfig()
    )
    paraphrase_defense_config: ParaphraseDefenseConfig = ParaphraseDefenseConfig()


@dataclass
class IterativeTrainingConfig:
    """
    Configs used in iterative (often adversarial) training.

    Attributes:
        iterative_training (bool): Whether to use iterative training.
        only_add_successful_adversarial_examples (bool):
            Whether to add strictly adversarial examples or not.
        min_num_new_examples_to_add (int):
            The minimum number of adversarial examples to add to
            the train set each attack round.
        max_num_search_for_adversarial_examples (int):
            The maximum number of examples to search for adversarial examples in
            each attack round. Think 'compute budget'.
        adversarial_example_search_minibatch_size (int):
            The size of the minibatches to use when searching for adversarial examples.
        num_iterative_training_rounds (int):
            The number of adversarial training rounds to do.
        use_probabilistic_robustness_check (bool):
            If true, only checks robustness on a random subset of the
            brute force attack dataset.
        skip_first_training_round (bool):
            Whether to skip the first training round or not.
        training_attack (AttackConfig):
            Config for the attack to use in adversarial training.
    """

    iterative_training: bool = False
    only_add_successful_adversarial_examples: bool = True
    min_num_new_examples_to_add: int = 50
    max_num_search_for_adversarial_examples: int = 8192
    adversarial_example_search_minibatch_size: int = 64
    num_iterative_training_rounds: int = 3
    use_probabilistic_robustness_check: bool = False
    skip_first_training_round: bool = False
    training_attack: AttackConfig = AttackConfig()


@dataclass
class EnvironmentConfig:
    """
    Configs used in environment setup (including dataset).

    Attributes:
        model_name_or_path (str): Either HF name or path to model checkpoint.
        decoder_name (Optional[str]): Decoder model name (used for defenses).
        is_pythia (bool) : Whether the architecture is Pythia or not. Needed
            for loading the model.
        dataset_type (str): Dataset type (tomita, tensor_trust).
        dataset_generation_style (str):
            How to generate the negative examples in the dataset.
            Only works with tensor trust for now.
        language_generator (str):
            Choose the regular language to use (tomita1, tomita2, tomita4, tomita7).
        max_length (int): The maximum length of the strings to generate.
        seed (int):
            The seed to use for the random number generator used to make the dataset.
        train_set_size (Optional[int]):
            The size of the train set.
            For generated datasets, must be set to positive integer.
            For HF datasets, can be set to None to use the full dataset.
        validation_set_size (Optional[int]):
            The size of the validation set.
            For generated datasets, must be set to positive integer.
            For HF datasets, can be set to None to use the full dataset.
        shuffle_train_set (bool):
            Whether to shuffle the train set. Can matter if we subsample.
        shuffle_validation_set (bool):
            Whether to shuffle the validation set. Can matter if we subsample.
    """

    model_name_or_path: str = "bert-base-uncased"
    decoder_name: Optional[str] = None
    is_pythia: bool = False
    dataset_type: str = "tomita"
    dataset_generation_style: str = (
        "random_words"  # random_word / random_character_edit
    )
    language_generator: str = "tomita4"
    max_length: int = 50
    seed: int = 0
    train_set_size: Optional[int] = None
    validation_set_size: Optional[int] = None
    shuffle_train_set: bool = False
    shuffle_validation_set: bool = False


@dataclass
class TrainingConfig:
    """Configs used across different training procedures.

    Attributes:
        iterative (IterativeTrainingConfig): Configs for iterative training.
        baseline (BaselineTrainingConfig): Configs for baseline training.
        num_train_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate to use in training.
        batch_size (int): Batch size to use in training.
        eval_steps (Optional[int | float]): Number of update steps between two
            evaluations. Will default to the same value as logging_steps if not set.
            Should be an integer or a float in range [0,1). If smaller than 1, will
            be interpreted as ratio of total training steps.
        logging_steps (int | float): Number of update steps between two logs. Should
            be an integer or a float in range [0,1). If smaller than 1, will be
            interpreted as ratio of total training steps.
        checkpoint (int): The checkpoint to start from.
        log_datasets_to_wandb (bool): Whether to log datasets to wandb. Off by default,
            as it takes a lot of space.
        model_save_path_prefix_or_hf (Optional[str]): Where to save the final
            checkpoint. If None, the model is not saved. If "hf", the model is saved to
            HuggingFace. Otherwise, the model is saved to a location starting with the
            specified prefix.

    For now, works only for the training pipeline.
    """

    iterative: IterativeTrainingConfig = IterativeTrainingConfig()
    baseline: BaselineTrainingConfig = BaselineTrainingConfig()
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    eval_steps: Optional[int] = None
    logging_steps: int | float = 500
    checkpoint: int = 142000
    log_datasets_to_wandb: bool = False
    model_save_path_prefix_or_hf: Optional[str] = SHARED_DATA_DIR
    force_name_to_save: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configs used in evaluation.

    Attributes:
        batch_size (int): The mini-batch size used to iterate over the dataset when
            evaluating.
        evaluation_attack (AttackConfig): Config for the attack to use in evaluation.
        num_generated_examples (Optional[int]): Number of adversarial examples to
            generate with the attack. Should be set iff the attack does not take dataset
            as an input.
        num_examples_to_log_detailed_info (Optional[int]): Number of adversarial
            examples for which we want to log detailed info, such as the original and
            attacked text, attack results and debug info. If None, do not log anything.
    """

    batch_size: int = 8
    evaluation_attack: AttackConfig = AttackConfig()
    num_generated_examples: Optional[int] = None
    num_examples_to_log_detailed_info: Optional[int] = 10


@dataclass
class ExperimentConfig:
    """
    Configs used in the experiment.

    Attributes:
        experiment_type (str): Type of experiment, from the ones defined in __main__.py.
        experiment_name (str): Name of the overarching experiment. Used to set a "group"
            in wandb. Each experiment can have several jobs.
        job_type (str): Name of the sub-experiment.
        run_name (str): Name of the individual run.
        environment (EnvironmentConfig): Configs for environment setup.
        training (TrainingConfig): Configs for training.
        evaluation (EvaluationConfig): Configs for evaluation.
        defense (DefenseConfig): Configs for defense setup.
    """

    experiment_type: str = MISSING
    experiment_name: str = "default-experiment"
    job_type: str = "default-job"
    run_name: str = "default-run"
    environment: EnvironmentConfig = EnvironmentConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    defense: DefenseConfig = DefenseConfig()


@dataclass
class OverallConfig:
    experiment: ExperimentConfig = ExperimentConfig()
