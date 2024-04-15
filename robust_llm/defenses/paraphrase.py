"""Text paraphrasing defense."""

from typing import Any, TypeAlias

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType

from robust_llm.defenses.defense import DefendedModel

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
        tokenizer: PreTrainedTokenizerBase,
        meta_prompt: str,
        temperature: float,
        paraphraser: PreTrainedModel,
        paraphraser_tokenizer: PreTrainedTokenizerBase,
        verbose: bool = False,
    ):
        self.tokenizer = tokenizer
        self.meta_prompt = meta_prompt
        self.temperature = temperature
        self.paraphraser = paraphraser
        self.paraphraser_tokenizer = paraphraser_tokenizer
        self.verbose = verbose

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length

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
        paraphrase_tokens = self.paraphraser_tokenizer(
            [self.meta_prompt + str(t) for t in text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        paraphrase_tokens.to(self.paraphraser.device)
        orig_len = paraphrase_tokens.input_ids.shape[1]  # original sequence length
        paraphrase = self.paraphraser.generate(
            **paraphrase_tokens,  # type: ignore
            temperature=self.temperature,
            max_new_tokens=orig_len,
            pad_token_id=self.paraphraser_tokenizer.pad_token_id,
            do_sample=True,
        )
        paraphrase = paraphrase[:, orig_len:]
        paraphrased_text = self.paraphraser_tokenizer.batch_decode(paraphrase)
        if self.verbose:
            print(f"Original text: {text}")
            print(f"Paraphrase: {paraphrased_text}")
        # Now tokenize the paraphrase
        return self.tokenizer(
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
    def __post_init__(self) -> None:
        super().__post_init__()
        self.paraphraser = AutoModelForCausalLM.from_pretrained(
            self.paraphraser_name
        ).to(self.device)
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained(self.paraphraser_name)
        self.paraphrase_tokenizer.padding_side = self.padding_side
        if "pad_token" not in self.paraphrase_tokenizer.special_tokens_map:
            self.paraphrase_tokenizer.pad_token = self.paraphrase_tokenizer.eos_token
        self.tokenizer = ParaphraseTokenizer(
            tokenizer=self.tokenizer,
            meta_prompt=self.meta_prompt,
            temperature=self.temperature,
            paraphraser=self.paraphraser,
            paraphraser_tokenizer=self.paraphrase_tokenizer,
            verbose=self.verbose,
        )

    @property
    def device(self) -> torch.device:
        return torch.device(self.defense_config.paraphrase_defense_config.device)

    @property
    def padding_side(self) -> str:
        return self.defense_config.paraphrase_defense_config.padding_side

    @property
    def verbose(self) -> bool:
        return self.defense_config.paraphrase_defense_config.verbose

    @property
    def paraphraser_name(self) -> str:
        return self.defense_config.paraphrase_defense_config.model_name

    @property
    def meta_prompt(self) -> str:
        return self.defense_config.paraphrase_defense_config.meta_prompt

    @property
    def temperature(self) -> float:
        return self.defense_config.paraphrase_defense_config.temperature

    def forward(self, **inputs) -> Any:
        return self.model(**inputs)
