from typing import Optional

import torch
import torch.utils.data
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from robust_llm.defenses.defense import DefendedModel
from robust_llm.utils import LanguageModel


def compute_perplexity(
    model: LanguageModel,
    window_size: Optional[int] = None,
    **inputs,
) -> torch.Tensor:
    """Compute the perplexity of the model on the given inputs.

    If `window_size` specified, will evaluate in contiguous chunks of `window_size`,
    dropping any tokens at the end of the string if `window_size` does not divide
    the number of tokens.

    Returns:
        A batch-shaped tensor of perplexities (negative log probs).
    """
    outputs = model(**inputs)
    logits = torch.log_softmax(outputs.logits, -1)  # [batch pos vocab]
    next_token_logits = torch.gather(
        logits, 2, inputs["input_ids"][:, 1:].unsqueeze(-1)
    ).squeeze(
        -1
    )  # [batch pos-1]
    mask = inputs["attention_mask"][:, 1:] == 1  # [batch pos-1]
    masked_next_token_logits = torch.where(
        mask, next_token_logits, torch.zeros_like(next_token_logits)
    )  # masked tokens have log prob 0 # [batch, pos-1]
    if window_size is not None:
        # chunk tokens into windows of size window_size
        num_windows = next_token_logits.shape[1] // window_size
        # TODO: handle case where window_size doesn't divide num_tokens
        masked_next_token_logits = masked_next_token_logits[
            :, : num_windows * window_size
        ]
        mask = mask[:, : num_windows * window_size]
        masked_next_token_logits = masked_next_token_logits.reshape(
            next_token_logits.shape[0], num_windows, window_size
        )  # [batch num_windows window_size]
        mask = mask.reshape(next_token_logits.shape[0], num_windows, window_size)
        masked_next_token_logits = (
            masked_next_token_logits.sum(dim=2) / mask.sum(dim=2).float()
        )  # [batch num_windows]
        perplexity = -masked_next_token_logits.min(dim=1).values  # batch
    else:
        perplexity = (
            -masked_next_token_logits.sum(dim=1) / mask.sum(dim=1).float()
        )  # [batch]
    return perplexity


def compute_max_perplexity(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    batch_size: int,
) -> float:
    """
    Computes the maximum perplexity of the model on the given dataset.
    This is useful for setting the perplexity threshold.

    Args:
        model: the model to evaluate
        dataset: the dataset to evaluate on
        batch_size: the batch size to use for evaluation
    Returns:
        The maximum perplexity (float).
    """
    dataset = dataset.with_format("torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False  # type: ignore
    )
    perplexities = []
    for batch in dataloader:
        encoded_input = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded_input.to(model.device)
        perplexity = compute_perplexity(
            model=model,
            window_size=None,
            **encoded_input,
        )
        perplexities.append(perplexity)
    return float(torch.cat(perplexities).max().item())


class PerplexityDefendedModel(DefendedModel):
    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.decoder is not None
        assert isinstance(self.device, torch.device)
        self.decoder.to(device=self.device)  # type: ignore
        threshold = self.defense_config.perplexity_defense_config.perplexity_threshold
        if threshold is None:
            assert self.dataset is not None
            threshold = compute_max_perplexity(
                self.decoder,
                self.tokenizer,
                self.dataset,
                self.defense_config.perplexity_defense_config.batch_size,
            )
        if self.verbose:
            print(f"Setting perplexity threshold to {threshold}")
        self.threshold = threshold

    @property
    def verbose(self) -> bool:
        return self.defense_config.perplexity_defense_config.verbose

    @property
    def window_size(self) -> Optional[int]:
        return self.defense_config.perplexity_defense_config.window_size

    @property
    def windowed(self) -> bool:
        return self.defense_config.perplexity_defense_config.windowed

    def forward(self, **inputs):
        assert self.decoder is not None
        perplexity = compute_perplexity(
            model=self.decoder,
            window_size=self.window_size,
            **inputs,
        )
        output = self.model(**inputs)
        output["filters"] = perplexity > self.threshold
        if self.verbose:
            print(
                f"Perplexity: {perplexity}. "
                f"Threshold: {self.threshold}. "
                f"Filter: {output['filters']}"
            )
        return output
