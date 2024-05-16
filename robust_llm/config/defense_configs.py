from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from robust_llm.config.model_configs import ModelConfig


@dataclass
class DefenseConfig:
    """
    Configs used in defense setup.

    Attributes:
        seed (int): Random seed for the defense.
        num_preparation_examples (Optional[int]):
            Number of examples to give the defense for preparation.
            If None, use the full train set.
    """

    seed: int = 0
    num_preparation_examples: Optional[int] = None


@dataclass
class PerplexityDefenseConfig(DefenseConfig):
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
        decoder (ModelConfig): Decoder model to use.
    """

    perplexity_threshold_proportion: float = 0.01
    window_size: int = 8
    report_max_perplexity: bool = True
    batch_size: int = 4
    verbose: bool = False
    save_perplexity_curves: bool = False
    decoder: ModelConfig = MISSING

    def __post_init__(self):
        assert 0 <= self.perplexity_threshold_proportion <= 1
        assert self.window_size > 0


@dataclass
class RetokenizationDefenseConfig(DefenseConfig):
    """
    Configs used in re-tokenization-based defenses.

    Attributes:
        drop_percentage (float): Percentage of the byte pair merges to drop.
        verbose (bool): Whether to print out the tokens of each example.
    """

    drop_percentage: float = 0.2
    verbose: bool = False

    def __post_init__(self):
        assert 0 <= self.drop_percentage <= 1


@dataclass
class ParaphraseDefenseConfig(DefenseConfig):
    """
    Configs used in paraphrase-based defenses.

    Attributes:
        paraphraser (ModelConfig): Model to use for paraphrasing.
        meta_prompt (str): Meta-prompt to use for paraphrasing.
        temperature (float): Temperature to use for generation.
        verbose (bool): Verbosity.
    """

    paraphraser: ModelConfig = ModelConfig(
        name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
        family="mistralai",
        revision="main",
        inference_type="generation",
        strict_load=True,
        padding_side="left",
    )
    meta_prompt: str = "Paraphrase the following sentences: "
    temperature: float = 0.7
    verbose: bool = False


cs = ConfigStore.instance()
cs.store(name="PERPLEXITY", group="defense", node=PerplexityDefenseConfig)
cs.store(name="RETOKENIZATION", group="defense", node=RetokenizationDefenseConfig)
cs.store(name="PARAPHRASE", group="defense", node=ParaphraseDefenseConfig)
