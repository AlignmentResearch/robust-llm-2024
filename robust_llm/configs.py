from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import MISSING

from robust_llm.attacks.text_attack.constants import TEXT_ATTACK_ATTACK_TYPES

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
        initial_kl_coefficient (float):
            The starting KL coefficient for TRL.
            NOTE: this might not be the optimal value.
            TODO(niki): vary this value.
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
    """

    batch_size: int = 64
    mini_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 10
    initial_kl_coefficient: float = 0

    adversary_base_model_name: str = "EleutherAI/pythia-14m"
    adversary_base_model_checkpoint: int = 143000

    min_length: int = -1
    max_new_tokens: int = 3


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
    gcg_attack_config: GCGAttackConfig = GCGAttackConfig()
    beam_search_attack_config: BeamSearchAttackConfig = BeamSearchAttackConfig()

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
            If the attack needs training, how often to train it.
            If None, only train attack after the first training round.
        victim_inference_batch_size (int):
            Batch size to use for victim model inference.
        append_to_modifiable_chunk: if False, the modifiable chunk is replaced by dummy
            attack tokens. Otherwise, dummy attack tokens are added after the original
            content of the modifiable chunk. Our attack then operates on these attack
            tokens.
        brute_force_tomita_attack_config (BruteForceTomitaAttackConfig):
            Config for BruteForceTomitaAttack.
        text_attack_attack_config (TextAttackAttackConfig):
            Config for TextAttackAttack.
        random_token_attack_config (RandomTokenAttackConfig):
            Config for RandomTokenAttack.
        trl_attack_config (TRLAttackConfig):
            Config for TRLAttack.
        gcg_attack_config (GCGAttackConfig):
            Config for GCGAttack.
    """

    attack_type: str = "identity"
    seed: int = 0
    train_frequency: Optional[int] = None
    victim_inference_batch_size: int = 8
    append_to_modifiable_chunk: bool = False

    # Configs for specific types of attacks.
    brute_force_tomita_attack_config: BruteForceTomitaAttackConfig = (
        BruteForceTomitaAttackConfig()
    )
    text_attack_attack_config: TextAttackAttackConfig = TextAttackAttackConfig()
    random_token_attack_config: RandomTokenAttackConfig = RandomTokenAttackConfig()
    trl_attack_config: TRLAttackConfig = TRLAttackConfig()
    search_based_attack_config: SearchBasedAttackConfig = SearchBasedAttackConfig()

    def __post_init__(self):
        if self.attack_type in TEXT_ATTACK_ATTACK_TYPES:
            assert not self.append_to_modifiable_chunk, "Not supported!"


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
        device (str): Device to use for models.
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
    optimizer: str = "adamw_torch"
    gradient_checkpointing: bool = False
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
            generate with the attack. Needs to be set if the attack does not take
            dataset as an input. If there is dataset, this option will limit the number
            of examples to attack.
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

    def __post_init__(self):
        # Ensure that the config is valid.
        evaluation_attack = self.evaluation.evaluation_attack

        if (
            self.environment.dataset_type == "tensor_trust"
            and evaluation_attack.attack_type == "search_based"
        ):
            # This is especially important for examples with True label; after wiping
            # out contents of modifiable chunk (the user's guess), the label changes
            # to False (which is desirable).
            assert (
                not evaluation_attack.append_to_modifiable_chunk
            ), "For tensor_trust, we want to forget about the original password guess."


@dataclass
class OverallConfig:
    experiment: ExperimentConfig = ExperimentConfig()
