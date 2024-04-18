from typing import Optional

import torch
import torch.utils.data
from datasets import Dataset
from tdigest import TDigest
from transformers import PreTrainedTokenizerBase

from robust_llm.defenses.defense import DefendedModel
from robust_llm.utils import LanguageModel


def compute_perplexity(
    model: LanguageModel,
    window_size: Optional[int] = None,
    **inputs,
) -> torch.Tensor:
    """Compute the perplexity of the model on the given inputs.

    If `window_size` is specified, this will evaluate in contiguous chunks
    of size `window_size`, dropping any tokens at the end of the string
    if `window_size` does not divide the number of tokens.

    Returns:
        A batch-shaped tensor of perplexities (negative log probs).
    """
    outputs = model(**inputs)

    # Softmax the logits so now the last dimension
    # is a probability distribution over the vocabulary

    # We think of the softmaxed logits as a probability distribution over all tokens,
    # which says how likely each token is to be the _next_ token.

    # start_token,    token_1,    token_2,    token_3,    token_4(=end_token)
    #        \           \           \           \           \
    #         \           \           \           \           \
    #          \           \           \           \           \
    #           \           \           \           \           \
    #            \           \           \           \           \
    #             \           \           \           \           \
    #              \      softmaxed logits predict     \           \
    #               \           \           \           \           \
    #                \           \           \           \           \
    #                 \           \           \           \           \
    #                  \           \           \           \           \
    #                   \           \           \           \           \
    #                    v           v           v           v           v
    #           dist_token_1, dist_token_2, dist_token_3, dist_token_4, ignore_this

    # Shape is [batch_size, num_tokens_per_datapoint, vocab_size]
    logits = torch.log_softmax(outputs.logits, -1)

    # We want to know how well the model predicted the next tokens,
    # for the "next tokens" that we actually saw. We do this by
    # getting the log probability mass of token_i in dist_token_i,
    # for each index i = 1, ..., num_tokens_per_datapoint - 1.

    # Note that `input_ids` has dimensions [batch_size, num_tokens_per_datapoint],
    # so we need to add a dimension to it to use it in `gather`.

    # Shape is [batch_size, num_tokens_per_datapoint - 1]
    # (no vocab_size dimension because we extracted the probability
    # corresponding to the token in question along that dimension)
    logits_without_final_prediction = logits[:, :-1, :]
    input_ids_without_initial_token = inputs["input_ids"][:, 1:].unsqueeze(-1)
    next_token_logits = torch.gather(
        input=logits_without_final_prediction,
        dim=2,
        index=input_ids_without_initial_token,
    ).squeeze(-1)

    # We ignore the probabilities of the padding tokens.
    # Shape is [batch_size, num_tokens_per_datapoint - 1]
    mask = inputs["attention_mask"][:, 1:] == 1
    masked_next_token_logits = torch.where(
        condition=mask,
        input=next_token_logits,
        other=torch.zeros_like(next_token_logits),
    )

    # TODO(niki): delete or document the window code
    # (this is addressed in PR #305)
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
        # Finally, sum the log probabilities of the next tokens
        # along the datapoint dimension, and divide by the number
        # of tokens in that datapoint.
        # Shape is [batch_size]
        denominator = mask.sum(dim=1).float()
        if torch.any(denominator < 1):
            raise ValueError("Can't take perplexity of empty sequences.")
        perplexity = -masked_next_token_logits.sum(dim=1) / mask.sum(dim=1).float()
    return perplexity


def compute_max_min_percentile_perplexity(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    batch_size: int,
) -> tuple[float, float, TDigest]:
    """
    Computes the maximum, minimum, and approximate percentile perplexities
    of the model on the given dataset.
    This is useful for setting the perplexity threshold.

    Args:
        model: the model to evaluate
        dataset: the dataset to evaluate on
        batch_size: the batch size to use for evaluation
    Returns:
        A tuple of the maximum perplexity, minimum perplexity, and a TDigest
        object from which approximate percentiles can be extracted.
    """
    dataset = dataset.with_format("torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False  # type: ignore
    )
    min_perplexity_so_far = float("inf")
    max_perplexity_so_far = 0.0
    tdigest = TDigest()
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
        min_perplexity_so_far = min(min_perplexity_so_far, perplexity.min().item())
        max_perplexity_so_far = max(max_perplexity_so_far, perplexity.max().item())
        tdigest.batch_update(perplexity)
    return max_perplexity_so_far, min_perplexity_so_far, tdigest


class PerplexityDefendedModel(DefendedModel):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.max_perplexity: Optional[float] = None
        self.min_perplexity: Optional[float] = None
        self.tdigest: Optional[TDigest] = None

        assert self.decoder is not None
        assert isinstance(self.device, torch.device)
        self.decoder.to(device=self.device)  # type: ignore
        # We subtract from 1 because perplexities are sorted low -> high,
        # and we want to get the x% _highest_ (not lowest) perplexity value.
        perplexity_config = self.defense_config.perplexity_defense_config
        proportion = perplexity_config.perplexity_threshold_proportion

        assert self.dataset is not None
        self.max_perplexity, self.min_perplexity, self.tdigest = (
            compute_max_min_percentile_perplexity(
                self.decoder,
                self.tokenizer,
                self.dataset,
                self.defense_config.perplexity_defense_config.batch_size,
            )
        )
        print("the max perplexity was", self.max_perplexity)
        print("the min perplexity was", self.min_perplexity)

        # TDigest takes percentages as input
        # Since we want to filter out `proportion` of the perplexities,
        # we take the (1 - proportion)th percentile.
        self.threshold = self.tdigest.percentile((1 - proportion) * 100)
        print(
            f"Setting perplexity threshold to {round(proportion * 100, 4)}% "
            f"(perplexity={round(self.threshold, 4)})"
        )

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
                f"Perplexity: {perplexity}\n"
                f"Threshold: {self.threshold}\n"
                f"Filter: {output['filters']}\n"
            )
        return output

    def get_all_perplexity_thresholds(self, dataset: Dataset) -> list[float]:
        """This method is used to get the perplexity thresholds for all percentiles.

        This is useful for exploring the perplexities of different models on
        both attacked and not-attacked datasets. Note that it doesn't touch the
        stored max_perplexity, min_perplexity, or tdigest.

        Args:
            dataset: the dataset to evaluate on

        Returns:
            A list of perplexity thresholds for percentiles [0%, 1%, ..., 100%]
        """
        _, _, tdigest = compute_max_min_percentile_perplexity(
            self.decoder,  # type: ignore
            self.tokenizer,
            dataset,
            self.defense_config.perplexity_defense_config.batch_size,
        )

        return [tdigest.percentile(p) for p in range(101)]
