import torch
import torch.utils.data
import wandb
from datasets import Dataset
from tdigest import TDigest
from transformers import PreTrainedTokenizerBase

from robust_llm.defenses.defense import DefendedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import LanguageModel


def _get_logits_in_windows(
    num_windows: int, window_size: int, masked_next_token_logits, side: str
) -> torch.Tensor:
    # Truncate the next token logits and mask to be divisible by window_size
    if side == "left":
        masked_next_token_logits = masked_next_token_logits[: num_windows * window_size]
    elif side == "right":
        masked_next_token_logits = masked_next_token_logits[
            -num_windows * window_size :
        ]
    else:
        raise ValueError("`side` must be 'left' or 'right'")

    return torch.reshape(masked_next_token_logits, (num_windows, window_size))


def _is_ones_then_zeros(mask):
    """Check that the mask is of the form [True, True, ..., False, False, ...]."""
    return (
        set(mask.unique().tolist()) <= {False, True}
        and torch.all(mask[: mask.sum().item()])
        and torch.all(~mask[mask.sum().item() :])
    )


def _get_perplexity_from_start_side(
    masked_next_token_logits: torch.Tensor,
    window_size: int,
    report_max_perplexity: bool,
    start_side: str,
) -> torch.Tensor:
    # Extract the subset of the next token logits and mask
    # such that we have a number of tokens that is divisible
    # by window_size
    # Shape is [num_windows window_size]
    num_windows = len(masked_next_token_logits) // window_size
    logits_in_windows = _get_logits_in_windows(
        num_windows=num_windows,
        window_size=window_size,
        masked_next_token_logits=masked_next_token_logits,
        side=start_side,
    )

    # Get the average perplexity of each window
    # Shape is [num_windows]
    window_average_logits = logits_in_windows.sum(dim=1) / window_size

    # Extract the maximum or average perplexity across windows.
    # Shape is []
    if report_max_perplexity:
        perplexity = torch.max(-window_average_logits, dim=0).values
    else:
        perplexity = torch.mean(-window_average_logits, dim=0)

    return perplexity


def _get_single_datapoint_perplexity(
    masked_next_token_logits: torch.Tensor,
    mask: torch.Tensor,
    window_size: int,
    report_max_perplexity: bool,
) -> torch.Tensor:
    # Now that we are only dealing with one datapoint, we
    # can cut off the masked tokens at the end
    assert _is_ones_then_zeros(mask)
    masked_next_token_logits = masked_next_token_logits[: int(mask.sum().item())]

    # If masked_next_token_logits is empty then there was at most one non-masked
    # token, so perplexity is not well-defined. We make the choice to throw an
    # error here, because a one-token sequence was almost certainly not intended.
    if len(masked_next_token_logits) == 0:
        raise ValueError(
            "Tried to calculate perplexity on a single token, which is undefined."
            " Does the dataset consist of a single ChunkType.OVERWRITABLE chunk?"
        )

    assert torch.all(masked_next_token_logits <= 0), (
        "The masked_next_token_logits should be nonpositive log probabilities, "
        f"but got a positive value:\n{masked_next_token_logits=}"
    )

    # Cut down the window size if it exceeds the number
    # of tokens we have in this example
    window_size = min(window_size, len(masked_next_token_logits))

    # For this single datapoint, we calculate
    # the perplexity across two sets of windows:
    # one that is flush with the left side, and one that is
    # flush with the right side. This way, we don't miss out
    # on high perplexity regions at the beginning or end of the
    # sequences. Below is an example.

    # Starting sequence:
    # [logit_token_1, logit_token_2, logit_token_3, logit_token_4,
    #  logit_token_5, logit_token_6, logit_token_7, logit_token_8]

    # If we have a window size of 3, starting from left side we would take
    # [logit_token_1, logit_token_2, logit_token_3] and
    # [logit_token_4, logit_token_5, logit_token_6];
    # while starting from the right side we would take
    # [logit_token_3, logit_token_4, logit_token_5] and
    # [logit_token_6, logit_token_7, logit_token_8].
    # We would do analogously for the mask.

    # We consider the perplexities across both of these sets of windows,
    # and take their maximum or average, depending on the value of
    # `report_max_perplexity`.
    start_left_perplexity = _get_perplexity_from_start_side(
        masked_next_token_logits=masked_next_token_logits,
        window_size=window_size,
        report_max_perplexity=report_max_perplexity,
        start_side="left",
    )
    start_right_perplexity = _get_perplexity_from_start_side(
        masked_next_token_logits=masked_next_token_logits,
        window_size=window_size,
        report_max_perplexity=report_max_perplexity,
        start_side="right",
    )
    stacked_perplexities = torch.stack([start_left_perplexity, start_right_perplexity])

    # Finally, take the average or maximum across the two sets of windows
    # starting from the left and right sides.
    # Shape is []
    if report_max_perplexity:
        perplexity = torch.max(stacked_perplexities, dim=0).values
    else:
        perplexity = torch.mean(stacked_perplexities, dim=0)

    return perplexity


def compute_perplexity(
    model: LanguageModel,
    model_inputs,  # TODO(niki): type
    window_size: int,
    report_max_perplexity: bool,
) -> torch.Tensor:
    """Compute the perplexity of the model on the given inputs.

    If `window_size` is specified, this will evaluate in contiguous chunks
    of size `window_size`, dropping any tokens at the end of the string
    if `window_size` does not divide the number of tokens.

    Args:
        model: the model to evaluate
        window_size: the size of the sliding window, if any
        window_stride: the stride of the window. Required if `window_size`
            is not None. Note that we always consider the rightmost
            window as well, even if the stride doesn't match up with it.
        report_max_perplexity: whether to report the maximum perplexity
            across windows, rather than the average perplexity.
        inputs: the inputs to the model

    Returns:
        A batch-shaped tensor of perplexities (negative log probs).
    """
    outputs = model(**model_inputs)

    # Softmax the logits so now the last dimension
    # is a probability distribution over the vocabulary

    # We think of the softmaxed logits as a probability distribution over all tokens,
    # which says how likely each token is to be the _next_ token. For example,
    # dist_token_2 is a probability distribution over all tokens, saying how
    # likely each token is to be the next token after token_1.

    # token_0,    token_1,    token_2,    token_3,    token_4
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
    logits = torch.log_softmax(outputs.logits, -1).detach()

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
    input_ids_without_initial_token = model_inputs["input_ids"][:, 1:].unsqueeze(-1)
    next_token_logits = torch.gather(
        input=logits_without_final_prediction,
        dim=2,
        index=input_ids_without_initial_token,
    ).squeeze(-1)

    # We ignore the probabilities of the padding tokens.
    # Shape is [batch_size, num_tokens_per_datapoint - 1]
    mask = model_inputs["attention_mask"][:, 1:] == 1
    masked_next_token_logits = torch.where(
        condition=mask,
        input=next_token_logits,
        other=torch.zeros_like(next_token_logits),
    )

    # Unfortunately, we need to calculate the perplexities for each
    # datapoint separately. We cannot do it in batch mode because
    # of the challenges arising from variable-length sequences.
    batch_perplexities = []
    for single_example_logits, mask in zip(masked_next_token_logits, mask):
        # Calculate the perplexity of each window
        perplexities = _get_single_datapoint_perplexity(
            masked_next_token_logits=single_example_logits,
            mask=mask,
            window_size=window_size,
            report_max_perplexity=report_max_perplexity,
        )
        batch_perplexities.append(perplexities)

    return torch.tensor(batch_perplexities)


def compute_max_min_percentile_perplexity(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    batch_size: int,
    window_size: int,
    report_max_perplexity: bool,
) -> tuple[float, float, TDigest]:
    """
    Computes the maximum, minimum, and approximate percentile perplexities
    of the model on the given dataset.
    This is useful for setting the perplexity threshold.

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
            model_inputs=encoded_input,
            window_size=window_size,
            report_max_perplexity=report_max_perplexity,
        )
        min_perplexity_so_far = min(min_perplexity_so_far, perplexity.min().item())
        max_perplexity_so_far = max(max_perplexity_so_far, perplexity.max().item())
        tdigest.batch_update(perplexity)
    return max_perplexity_so_far, min_perplexity_so_far, tdigest


class PerplexityDefendedModel(DefendedModel):
    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.decoder is not None
        assert isinstance(self.device, torch.device)
        self.decoder.to(device=self.device)  # type: ignore
        self.window_size = self.defense_config.perplexity_defense_config.window_size
        self.report_max_perplexity = (
            self.defense_config.perplexity_defense_config.report_max_perplexity
        )
        self.batch_size = self.defense_config.perplexity_defense_config.batch_size
        self.verbose = self.defense_config.perplexity_defense_config.verbose

        # We subtract from 1 because perplexities are sorted low -> high,
        # and we want to get the x% _highest_ (not lowest) perplexity value.
        perplexity_config = self.defense_config.perplexity_defense_config
        proportion = perplexity_config.perplexity_threshold_proportion

        assert self.dataset is not None
        self.max_perplexity, self.min_perplexity, self.tdigest = (
            compute_max_min_percentile_perplexity(
                model=self.decoder,
                tokenizer=self.tokenizer,
                dataset=self.dataset,
                batch_size=self.batch_size,
                window_size=self.window_size,
                report_max_perplexity=self.report_max_perplexity,
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

        # Don't log to wandb if we're running tests
        if wandb.run is not None:
            wandb.log(
                {
                    "defense/perplexity_threshold_proportion_pre_attack": (
                        perplexity_config.perplexity_threshold_proportion
                    ),
                    "defense/perplexity_threshold_value": self.threshold,
                    "defense/max_perplexity_pre_attack": self.max_perplexity,
                    "defense/min_perplexity_pre_attack": self.min_perplexity,
                },
                commit=False,
            )

    def forward(self, **inputs):
        assert self.decoder is not None
        perplexity = compute_perplexity(
            model=self.decoder,
            model_inputs=inputs,
            window_size=self.window_size,
            report_max_perplexity=self.report_max_perplexity,
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

    def get_all_perplexity_thresholds(
        self, dataset: RLLMDataset, text_column_to_use: str
    ) -> list[float]:
        """This method is used to get the perplexity thresholds for all percentiles.

        This is useful for exploring the perplexities of different models on
        both attacked and not-attacked datasets. Note that it doesn't touch the
        stored max_perplexity, min_perplexity, or tdigest.

        Args:
            dataset: the dataset to evaluate on
            text_column_to_use: the column of the dataset that should
                be used for the perplexity calculation
        Returns:
            A list of perplexity thresholds for percentiles [0%, 1%, ..., 100%]
        """
        # We don't pass in input_ids or attention_mask here because
        # compute_max_min_percentile_perplexity gets confused by them.
        hf_dataset_only_text = Dataset.from_dict(
            {"text": dataset.ds[text_column_to_use]}
        )

        assert self.decoder is not None
        _, _, tdigest = compute_max_min_percentile_perplexity(
            model=self.decoder,
            tokenizer=self.tokenizer,
            dataset=hf_dataset_only_text,
            batch_size=self.batch_size,
            window_size=self.window_size,
            report_max_perplexity=self.report_max_perplexity,
        )

        return [tdigest.percentile(p) for p in range(101)]
