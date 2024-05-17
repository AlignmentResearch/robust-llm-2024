"""Text paraphrasing defense."""

from typing import Any, TypeAlias

from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType

from robust_llm import logger
from robust_llm.config.defense_configs import DefenseConfig, ParaphraseDefenseConfig
from robust_llm.defenses.defense import DefendedModel
from robust_llm.models import WrappedModel

TextOrTokenSeqInput: TypeAlias = TextInput | PreTokenizedInput | list[PreTokenizedInput]


class ParaphraseTokenizer(PreTrainedTokenizerBase):
    """Paraphrase input text then tokenize.

    This tokenizer is used by the `ParaphraseDefendedModel` class and
    wraps around the original `transformers.PreTrainedTokenizerBase` object.
    It first generates a paraphrase of the input text using a paraphraser,
    then tokenizes the paraphrase using the original tokenizer.
    """

    def __init__(
        self,
        victim_tokenizer: PreTrainedTokenizerBase,
        meta_prompt: str,
        temperature: float,
        paraphraser: WrappedModel,
        verbose: bool = False,
    ):
        self.victim_tokenizer = victim_tokenizer
        self.meta_prompt = meta_prompt
        self.temperature = temperature
        self.paraphraser = paraphraser
        self.verbose = verbose

    @property
    def model_max_length(self) -> int:
        return self.victim_tokenizer.model_max_length

    # pyright complains because we don't have a setter for this property
    # while the base class does, but we really don't need one so we ignore
    # the error.
    @property
    def pad_token_id(self) -> int | None:  # type: ignore[reportIncompatibleMethodOverride]  # noqa: E501
        return self.victim_tokenizer.pad_token_id

    def __call__(
        self,
        text: TextOrTokenSeqInput | None = None,
        text_pair: TextOrTokenSeqInput | None = None,
        text_target: TextOrTokenSeqInput | None = None,
        text_pair_target: TextOrTokenSeqInput | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Generate paraphrase using temperature, meta-prompt and paraphraser
        assert text is not None
        if isinstance(text, TextInput):
            text = [text]
        assert all(isinstance(t, str) for t in text)
        paraphrase_tokens = self.paraphraser.tokenizer(
            [self.meta_prompt + str(t) for t in text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        paraphrase_tokens.to(self.paraphraser.device)
        orig_len = paraphrase_tokens.input_ids.shape[1]  # original sequence length
        paraphrase = self.paraphraser.model.generate(
            **paraphrase_tokens,  # type: ignore
            temperature=self.temperature,
            max_new_tokens=orig_len,
            pad_token_id=self.paraphraser.tokenizer.pad_token_id,
            do_sample=True,
        )
        paraphrase = paraphrase[:, orig_len:]
        paraphrased_text = self.paraphraser.tokenizer.batch_decode(paraphrase)
        if self.verbose:
            logger.debug("Original text: %s", text)
            logger.debug("Paraphrase: %s", paraphrased_text)
        # Now tokenize the paraphrase
        return self.victim_tokenizer(
            paraphrased_text,
            text_pair,
            text_target,  # type: ignore
            text_pair_target,
            add_special_tokens,
            padding,
            truncation,  # type: ignore
            max_length,
            stride,
            is_split_into_words,
            pad_to_multiple_of,
            return_tensors,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose,
            **kwargs,
        )


class ParaphraseDefendedModel(DefendedModel):
    def __init__(
        self, victim: WrappedModel, defense_config: ParaphraseDefenseConfig
    ) -> None:
        super().__init__(victim)
        self.cfg = defense_config
        self.paraphraser = WrappedModel.from_config(
            config=self.cfg.paraphraser,
            accelerator=victim.accelerator,
        )
        if "pad_token" not in self.paraphraser.tokenizer.special_tokens_map:
            self.paraphraser.tokenizer.pad_token = self.paraphraser.tokenizer.eos_token

        self.paraphrase_tokenizer = ParaphraseTokenizer(
            victim_tokenizer=victim.tokenizer,
            meta_prompt=self.cfg.meta_prompt,
            temperature=self.cfg.temperature,
            paraphraser=self.paraphraser,
            verbose=self.cfg.verbose,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.paraphrase_tokenizer

    @property
    def defense_config(self) -> DefenseConfig:
        return self.cfg

    def forward(self, **inputs) -> Any:
        return self._underlying_model(**inputs)
