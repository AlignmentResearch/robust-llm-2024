from dataclasses import dataclass
from typing import Any, Mapping, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.utils.data
from datasets import Dataset
from transformers import PreTrainedModel

from robust_llm.configs import DefenseConfig, PerplexityDefenseConfig
from robust_llm.defenses.perplexity import (
    PerplexityDefendedModel,
    compute_max_perplexity,
    compute_perplexity,
)
from robust_llm.defenses.retokenization import (
    RetokenizationDefendedModel,
    broken_token_representations,
)


@dataclass
class TokenizedInput(Mapping):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> "TokenizedInput":
        return TokenizedInput(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


@dataclass
class Output:
    logits: torch.Tensor
    filters: Optional[torch.Tensor] = None

    # support item assignment like output['filters'] = ...
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


class TrivialTestDataset(torch.utils.data.Dataset):
    """Simple Dataset for testing."""

    def __init__(self):
        pass

    def __getitem__(self, index):
        # For each sample, we return a dict with a 'text' field
        return {
            "text": ["This is a test.", "Of two parts"][index],
            "input_ids": torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)[index],
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)[
                index
            ],
        }

    def __len__(self):
        return 2

    def set_format(self, *args, **kwargs):
        pass

    def with_format(self, *args, **kwargs):
        return self


@pytest.mark.parametrize(
    "input_ids, attention_mask, broken_input_ids, broken_attention_mask",
    [
        (
            [7, 1103, 8, 1188],
            [1, 1, 1, 1],
            [7, 24438, 1162, 8, 3947, 11798],
            [1, 1, 1, 1, 1, 1],
        ),
        (
            [[7, 1103, 8], [1188, 3, 4]],
            [[1, 1, 1], [1, 1, 1]],
            [[7, 24438, 1162, 8], [3947, 11798, 3, 4]],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ),
        (
            torch.tensor([7, 1103, 8, 1188]),
            torch.tensor([1, 1, 1, 1]),
            torch.tensor([7, 24438, 1162, 8, 3947, 11798]),
            torch.tensor([1, 1, 1, 1, 1, 1]),
        ),
        (
            torch.tensor([[7, 1103, 8], [1188, 3, 4]]),
            torch.tensor([[1, 1, 1], [1, 1, 1]]),
            torch.tensor([[7, 24438, 1162, 8], [3947, 11798, 3, 4]]),
            torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
        ),
        (
            torch.tensor([7, 1103, 8, 9999]),
            torch.tensor([1, 1, 1, 1]),
            torch.tensor([7, 24438, 1162, 8, 9999]),
            torch.tensor([1, 1, 1, 1, 1]),
        ),
    ],
)
def test_broken_token_representations(
    input_ids, attention_mask, broken_input_ids, broken_attention_mask
):
    broken_tokens = [(1103, [24438, 1162]), (1188, [3947, 11798])]
    output_ids, output_mask = broken_token_representations(
        input_ids, attention_mask, broken_tokens, max_length=6
    )
    if isinstance(output_ids, torch.Tensor):
        assert isinstance(output_mask, torch.Tensor)
        assert torch.equal(output_ids, broken_input_ids)
        assert torch.equal(output_mask, broken_attention_mask)
    else:
        assert output_ids == broken_input_ids
        assert output_mask == broken_attention_mask


def test_compute_perplexity():
    # Step 1: Setup
    # Define a Mock model
    mock_model = MagicMock()

    # Define what mock_model should return when called
    output = torch.tensor(
        [
            [[-np.inf, 0.0], [-np.inf, 0.0], [-np.inf, 0.0]],
            [[0.0, -np.inf], [0.0, -np.inf], [0.0, -np.inf]],
            [[-np.inf, 0.0], [-np.inf, 0], [0.0, -np.inf]],
            [
                [-np.log(2), -np.log(2)],
                [-np.log(2), -np.log(2)],
                [-np.log(2), -np.log(2)],
            ],
            [
                [-np.log(2), -np.log(2)],
                [-np.log(2), -np.log(2)],
                [-np.log(2), -np.log(2)],
            ],
            [[-np.inf, 0.0], [-np.inf, 0.0], [-np.inf, 0.0]],
        ],
        dtype=torch.float,
    )
    mock_model.return_value.logits = output

    # Inputs for the function
    # "input_ids" and "attention_mask" are both torch.Tensor
    inputs = {
        "input_ids": torch.tensor(
            [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]],
            dtype=torch.long,
        ),
    }

    # Step 2: Call function
    output = compute_perplexity(mock_model, window_size=None, **inputs)

    # Step 3: Assert the output is as expected
    expected_output = torch.tensor(
        [np.inf, np.inf, 0.0, np.log(2), np.log(2), 0.0], dtype=torch.float
    )
    assert torch.allclose(
        output, expected_output
    ), f"Expected {expected_output}, but got {output}"


def test_compute_max_perplexity():
    # Step 1: Setup
    # Define a Mock model
    mock_model = MagicMock()
    mock_model_output = torch.tensor(
        [
            [
                [-np.log(3), -np.log(3), -np.log(3)],
                [-np.log(3), -np.log(3), -np.log(3)],
                [-np.log(3), -np.log(3), -np.log(3)],
            ],
        ],
        dtype=torch.float,
    )  # log of probabilities
    mock_model.return_value.logits = mock_model_output
    mock_model.device = "cpu"

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = TokenizedInput(
        input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
    )

    # Define a Mock dataset
    mock_dataset = TrivialTestDataset()

    # Step 2: Call function
    max_perplexity = compute_max_perplexity(
        mock_model, mock_tokenizer, mock_dataset, batch_size=1  # type: ignore
    )

    # Step 3: Assert the output is as expected
    expected_output = np.log(3)
    assert np.isclose(
        max_perplexity, expected_output
    ), f"Expected {expected_output}, but got {max_perplexity}"


def test_retokenization_defended_model_forward():
    # Create a dummy input
    inputs = {
        "input_ids": torch.tensor([[1, 1, 2], [1, 1, 0], [2, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0]]),
    }
    broken_tokens = [(2, [3, 4])]

    # Create a dummy dataset
    dataset = Dataset.from_dict(
        {
            "input_ids": [[0, 1, 2]],
            "attention_mask": [[1, 1, 1]],
        }
    )

    defense_config = DefenseConfig()

    init_model = MagicMock()
    tokenizer = MagicMock()
    decoder = MagicMock()

    tokenizer.model_max_length = 4
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = 0

    init_model.device = "cpu"
    decoder.device = "cpu"

    def side_effect_function(*args, **kwargs):
        return Output(
            logits=torch.tensor(
                [
                    [-np.inf, 0] if 3 in kwargs["input_ids"][i] else [0, -np.inf]
                    for i in range(len(kwargs["input_ids"]))
                ]
            )
        )

    init_model.side_effect = side_effect_function

    # Create a RetokenizationDefendedModel instance
    model = RetokenizationDefendedModel(
        init_model=init_model,
        tokenizer=tokenizer,
        decoder=decoder,
        dataset=dataset,
        defense_config=defense_config,
    )
    model.broken_tokens = broken_tokens

    # Call the forward method
    output = model.forward(**inputs)

    assert (
        output.logits
        == torch.tensor(
            [
                [-np.inf, 0],
                [0, -np.inf],
                [-np.inf, 0],
            ]
        )
    ).all()
    assert output.filters is None


def test_perplexity_defended_model_forward():
    # Create a dummy input
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2], [2, 1, 0], [0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    }

    # Create a dummy dataset
    dataset = Dataset.from_dict(
        {
            "input_ids": [[0, 1, 2]],
            "attention_mask": [[1, 1, 1]],
        }
    )

    # Create a dummy defense config
    perplexity_config = PerplexityDefenseConfig()
    perplexity_config.perplexity_threshold = 0.0
    defense_config = DefenseConfig(
        perplexity_defense_config=perplexity_config,
    )

    init_model = MagicMock()
    tokenizer = MagicMock()
    decoder = MagicMock()
    decoder.__class__ = PreTrainedModel

    init_model.device = torch.device("cpu")
    decoder.device = torch.device("cpu")

    output_logits = torch.tensor(
        [
            [-np.inf, 0],
            [0, -np.inf],
            [-np.inf, 0],
        ]
    )
    init_model.return_value = Output(
        logits=torch.tensor(output_logits),
    )
    decoder.return_value = Output(
        logits=torch.tensor(
            [
                [[-np.inf, 0, -np.inf], [-np.inf, -np.inf, 0], [0, 0, 0]],
                [[-np.inf, 0, -np.inf], [-np.inf, -np.inf, 0], [0, 0, 0]],
                [[-np.inf, 0, -np.inf], [-np.inf, -np.inf, 0], [0, 0, 0]],
            ]
        )
    )

    # Create a PerplexityDefendedModel instance
    model = PerplexityDefendedModel(
        init_model=init_model,
        tokenizer=tokenizer,
        decoder=decoder,
        dataset=dataset,
        defense_config=defense_config,
    )

    # Call the forward method
    output = model.forward(**inputs)

    assert (output.filters == torch.tensor([False, True, False])).all()
    assert (output.logits == output_logits).all()
