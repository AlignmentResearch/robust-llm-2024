from dataclasses import dataclass, field
from typing import Literal, Optional

import omegaconf
import torch
from omegaconf import MISSING

from robust_llm.attacks.trl.constants import TRL_REWARD_TYPES

SHARED_DATA_DIR = "/robust_llm_data"
ModelFamily = Literal["gpt2", "pythia", "bert"]


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
            modified by the attack. Otherwise, content is not modified at the
            start and the attack performs modifications on the original text.
            If None, all modifiable chunks should be PERTURBABLE.
            If int, all modifiable chunks should be OVERWRITABLE.
            TODO (GH#353): Refactor to remove the option for None.

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
        max_iterations (int): Maximum number of iterations to run the attack.
        logging_frequency (int): How often to log the attack.
        batch_size (int): Batch size to use for the victim pipeline used
            to check whether the attack was successful.
    """

    min_tokens: int = 1
    max_tokens: int = 3
    max_iterations: int = 100
    logging_frequency: int = 10
    batch_size: int = 8


@dataclass
class TRLAttackConfig:
    """Options specific for TRL attacks.

    Attributes:
        batch_size (int):
            The TRL batch size (how many examples are passed
            to a single call of PPO's "step" function).
        mini_batch_size (int):
            The TRL minibatch size (how many examples to load
            onto the gpu at once).
        gradient_accumulation_steps (int):
            The TRL gradient accumulation steps (how many minibatches
            to accumulate before taking a single gradient update step).
        ppo_epochs (int):
            The number of ppo epochs to run TRL on the provided dataset.
        adversary_base_model_name (str):
            Which model to use as the adversary.
        adversary_base_model_checkpoint (int):
            Which checkpoint to use for the adversary model.
        min_length (int):
            The minimum number of tokens to generate.
            If -1, there is no minimum length.
            Name and convention copied from trl code.
        max_new_tokens (int):
            The maximum number of tokens to generate.
            Name copied from trl code.
        reward_type (str):
            determines reward funtion to use in TRL.
        model_name_to_save (str):
            The name to use for saving the model.
        model_save_path_prefix (Optional[str]): Where to save the final
            checkpoint. If None, the model is not saved.
            Otherwise, the model is saved to a location starting with the
            specified prefix.
    """

    batch_size: int = 128
    mini_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 10

    adversary_base_model_name: str = "EleutherAI/pythia-14m"
    adversary_base_model_checkpoint: int = 143000

    min_length: int = -1
    max_new_tokens: int = 3

    reward_type: str = "minus_correct_logit_plus_incorrect_logits"

    model_name_to_save: str = "trl"
    model_save_path_prefix: Optional[str] = SHARED_DATA_DIR

    def __post_init__(self):
        assert (
            self.batch_size == self.mini_batch_size * self.gradient_accumulation_steps
        )

        assert self.reward_type in TRL_REWARD_TYPES


@dataclass
class GCGAttackConfig:
    """Required options with defaults for the GCG attack.

    Args:
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    top_k: int = 256


@dataclass
class BeamSearchAttackConfig:
    """Required options with defaults for the beam search attack.

    Args:
        beam_search_width: how many candidates to keep after each iteration.
    """

    beam_search_width: int = 5


@dataclass
class SearchBasedAttackConfig:
    """
    Required options with defaults for the search based attack.

    Args:
        n_candidates_per_it: the total number of token replacements
            to consider in each iteration (in GCG, this must be less than
            top_k * n_attack_tokens, which is the total number of candidates).
        n_its: total number of iterations to run
        n_attack_tokens: number of attack tokens to optimize
        forward_pass_batch_size: batch size used for forward pass when evaluating
            candidates. If None, defaults to n_candidates_per_it.
        seq_clf: whether we are using a SequenceClassification model
            (default alternative is a CausalLM)
    """

    n_candidates_per_it: int = 512
    n_its: int = 50
    n_attack_tokens: int = 10
    forward_pass_batch_size: Optional[int] = None
    seq_clf: bool = True
    search_type: str = "gcg"  # We currently support "gcg" and "beam_search"
    gcg_attack_config: GCGAttackConfig = field(default_factory=GCGAttackConfig)
    beam_search_attack_config: BeamSearchAttackConfig = field(
        default_factory=BeamSearchAttackConfig
    )

    def __post_init__(self):
        if (
            self.search_type == "gcg"
            and self.n_candidates_per_it
            > self.gcg_attack_config.top_k * self.n_attack_tokens
        ):
            raise ValueError(
                "n_candidates_per_it must be at most top_k * suffix_length"
            )

        if self.forward_pass_batch_size is None:
            self.forward_pass_batch_size = self.n_candidates_per_it
        elif self.forward_pass_batch_size > self.n_candidates_per_it:
            raise ValueError(
                "forward_pass_batch_size must be at most n_candidates_per_it"
            )


@dataclass
class AttackConfig:
    """
    Configs used in attack setup.

    Attributes:
        attack_type (str): The type of attack to use.
        seed (int): Random seed for the attack.
        train_frequency (Optional[int]):
            If the attack needs training, how often to train it,
            counted in number of victim iterative training rounds.
            If None, only train attack after the first victim training round.
            Must be positive or None.
        log_frequency (Optional[int]):
            If the attack needs training, how often to log training progress.
            If None, no training progress is logged. Must be positive or None.
        victim_inference_batch_size (int):
            Batch size to use for victim model inference.
        brute_force_tomita_attack_config (BruteForceTomitaAttackConfig):
            Config for BruteForceTomitaAttack.
        text_attack_attack_config (TextAttackAttackConfig):
            Config for TextAttackAttack.
        random_token_attack_config (RandomTokenAttackConfig):
            Config for RandomTokenAttack.
        trl_attack_config (TRLAttackConfig):
            Config for TRLAttack.
        search_based_attack_config (SearchBasedAttackConfig):
            Config for SearchBasedAttack.
    """

    attack_type: str = "identity"
    seed: int = 0
    train_frequency: Optional[int] = None
    log_frequency: Optional[int] = 1
    victim_inference_batch_size: int = 8

    # Configs for specific types of attacks.
    brute_force_tomita_attack_config: BruteForceTomitaAttackConfig = field(
        default_factory=BruteForceTomitaAttackConfig
    )
    text_attack_attack_config: TextAttackAttackConfig = field(
        default_factory=TextAttackAttackConfig
    )
    random_token_attack_config: RandomTokenAttackConfig = field(
        default_factory=RandomTokenAttackConfig
    )
    trl_attack_config: TRLAttackConfig = field(default_factory=TRLAttackConfig)
    search_based_attack_config: SearchBasedAttackConfig = field(
        default_factory=SearchBasedAttackConfig
    )

    def __post_init__(self):
        if self.train_frequency is not None and self.train_frequency <= 0:
            raise ValueError("train_frequency must be positive or None.")

        if self.log_frequency is not None and self.log_frequency <= 0:
            raise ValueError("log_frequency must be positive or None.")


@dataclass
class PerplexityDefenseConfig:
    """
    Configs used in perplexity-based defenses.

    Attributes:
        perplexity_threshold_proportion (float):
            What proportion of the train set should be filtered out by
            the perplexity filter? Must be between 0 and 1 inclusive.
        window_size (int): Window size. If the window is larger
            than a given example, it will be cut down to the
            length of that example for the perplexity calculation
            on that specific example. Thus, if you want to calculate
            perplexity over the entire example for every example,
            choose a window size that is larger than any example.
        report_max_perplexity (bool): Whether to report the maximum
            perplexity across windows, instead of the average.
        batch_size (int): Batch size to use for perplexity calculations.
        verbose (bool): Whether to print out the perplexity of each example.
        save_perplexity_curves (bool):
            Whether or not to save the full perplexity curves (what proportion
            of the dataset is filtered out at each perplexity value) for the
            original dataset and the attacked dataset.
    """

    perplexity_threshold_proportion: float = 0.01
    window_size: int = 8
    report_max_perplexity: bool = True
    batch_size: int = 4
    verbose: bool = False
    save_perplexity_curves: bool = False

    def __post_init__(self):
        assert 0 <= self.perplexity_threshold_proportion <= 1
        assert self.window_size > 0


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
        num_preparation_examples (Optional[int]):
            Number of examples to give the defense for preparation.
            If None, use the full train set.
        perplexity_defense_config (PerplexityDefenseConfig):
            Configs for perplexity-based defenses.
        retokenization_defense_config (RetokenizationDefenseConfig):
            Configs for re-tokenization-based defenses.
        paraphrase_defense_config (ParaphraseDefenseConfig):
            Configs for paraphrase-based defenses.
    """

    defense_type: str = "identity"
    seed: int = 0
    num_preparation_examples: Optional[int] = None
    perplexity_defense_config: PerplexityDefenseConfig = field(
        default_factory=PerplexityDefenseConfig
    )
    retokenization_defense_config: RetokenizationDefenseConfig = field(
        default_factory=RetokenizationDefenseConfig
    )
    paraphrase_defense_config: ParaphraseDefenseConfig = field(
        default_factory=ParaphraseDefenseConfig
    )


@dataclass
class IterativeTrainingConfig:
    """
    Configs used in iterative (often adversarial) training.

    Attributes:
        iterative_training (bool): Whether to use iterative training.
        num_examples_to_generate_each_round (int): The number of adversarial examples to
            generate each round for training.
        num_examples_to_log_to_wandb_each_round (int): The number of adversarial
            examples to log to wandb each round.
        only_add_successful_adversarial_examples (bool):
            Whether to add only successful adversarial examples to training set;
            otherwise, add all trials, successful or not.
        num_iterative_training_rounds (int):
            The number of adversarial training rounds to do.
        skip_first_training_round (bool):
            Whether to skip the first training round or not.
        training_attack (AttackConfig):
            Config for the attack to use in adversarial training.
    """

    iterative_training: bool = False
    num_examples_to_generate_each_round: int = 500
    num_examples_to_log_to_wandb_each_round: int = 10
    only_add_successful_adversarial_examples: bool = False
    num_iterative_training_rounds: int = 3
    skip_first_training_round: bool = False
    use_balanced_sampling: bool = False
    training_attack: AttackConfig = field(default_factory=AttackConfig)


@dataclass
class DatasetConfig:
    """Config used for dataset setup.

    Attributes:
        dataset_type (str): Type of dataset to use.
        n_train (int): Number of training examples.
        n_val (int): Number of validation examples.
        config_name (Optional[str]): config_name from hf datasets (if applicable).
        revision (str): The huggingface revision to start from. Defaults
            to <1.0.0 to avoid unexpected breaking changes.
        inference_type (str): The type of inference performed ("classification"
            or "generation")
    """

    dataset_type: str = omegaconf.MISSING
    n_train: int = 0
    n_val: int = 0
    config_name: Optional[str] = None
    revision: str = "<1.0.0"
    inference_type: str = "classification"


@dataclass
class EnvironmentConfig:
    """
    Configs used in environment setup.

    Attributes:
        model_name_or_path (str): Either HF name or path to model checkpoint.
        model_family (str) : Which model family to load (e.g. "gpt2").
        decoder_name (Optional[str]): Decoder model name (used for defenses).
        decoder_family (Optional[str]): Which model family the decoder belongs to.
        decoder_revision (Optional[str]): The revision of the decoder model.
        device (str): Device to use for models.
        test_mode (bool): Whether or not we're currently testing
    """

    model_name_or_path: str = "bert-base-uncased"
    model_family: str = "gpt2"
    decoder_name: Optional[str] = None
    decoder_family: Optional[str] = None
    decoder_revision: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_mode: bool = False


def __post_init__(self):
    if self.decoder_name is not None:
        assert self.decoder_family is not None
        assert self.decoder_revision is not None


@dataclass
class TrainingConfig:
    """Configs used across different training procedures.

    Attributes:
        iterative (IterativeTrainingConfig): Configs for iterative training.
        baseline (BaselineTrainingConfig): Configs for baseline training.
        num_train_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate to use in training.
        batch_size (int): Batch size to use in training (PER DEVICE!).
        eval_steps (Optional[int | float]): Number of update steps between two
            evaluations. Will default to the same value as logging_steps if not set.
            Should be an integer or a float in range [0,1). If smaller than 1, will
            be interpreted as ratio of total training steps.
        logging_steps (int | float): Number of update steps between two logs. Should
            be an integer or a float in range [0,1). If smaller than 1, will be
            interpreted as ratio of total training steps.
        save_strategy (str): The checkpoint save strategy to adopt during training.
            Possible values are:
            - "no": No save is done during training.
            - "epoch": Save is done at the end of each epoch.
            - "steps": Save is done every save_steps.
        save_steps (int | float): Number of updates steps before two checkpoint saves if
            save_strategy="steps". Should be an integer or a float in range [0,1). If
            smaller than 1, will be interpreted as ratio of total training steps.
        revision (str): The huggingface revision to start from.
        log_full_datasets_to_wandb (bool): Whether to log full datasets to wandb. Off
            by default, as it takes a lot of space.
        model_save_path_prefix_or_hf (Optional[str]): Where to save the final
            checkpoint. If None, the model is not saved. If "hf", the model is saved to
            HuggingFace. Otherwise, the model is saved to a location starting with the
            specified prefix.
        seed: seed to use for training. It will be set at the beginning of huggingface
            Trainer's training. In particular, it may affect random initialization
            (if any).

    For now, works only for the training pipeline.
    """

    iterative: IterativeTrainingConfig = field(default_factory=IterativeTrainingConfig)
    baseline: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    optimizer: str = "adamw_torch"
    gradient_checkpointing: bool = False
    eval_steps: Optional[int] = None
    logging_steps: int | float = 500
    save_strategy: str = "steps"
    save_steps: int | float = 500
    revision: str = "main"
    log_full_datasets_to_wandb: bool = False
    model_save_path_prefix_or_hf: Optional[str] = SHARED_DATA_DIR
    force_name_to_save: Optional[str] = None
    seed: int = 0

    def __post_init__(self):
        assert self.num_train_epochs > 0, "Number of training epochs must be positive."


@dataclass
class EvaluationConfig:
    """Configs used in evaluation.

    Attributes:
        batch_size (int): The mini-batch size used to iterate over the dataset when
            evaluating (PER DEVICE!).
        evaluation_attack (AttackConfig): Config for the attack to use in evaluation.
        num_examples_to_log_detailed_info (Optional[int]): Number of adversarial
            examples for which we want to log detailed info, such as the original and
            attacked text, attack results and debug info. If None, do not log anything.
    """

    batch_size: int = 8
    evaluation_attack: AttackConfig = field(default_factory=AttackConfig)
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
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = omegaconf.MISSING
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    defense: Optional[DefenseConfig] = None


@dataclass
class OverallConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
