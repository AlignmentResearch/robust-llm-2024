from dataclasses import dataclass
from unittest.mock import MagicMock

import torch
from transformers import PreTrainedTokenizerBase

from robust_llm.config.callback_configs import AutoregressiveCallbackConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.scoring_callbacks import (
    BinaryCallbackOutput,
    CallbackInput,
    build_binary_scoring_callback,
)


@dataclass
class MagicTokenized:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


tokenizer = MagicMock()
tokenizer.return_value = MagicTokenized(
    input_ids=torch.tensor([[0], [1]]),
    attention_mask=torch.tensor([[1], [1]]),
)
tokenizer.batch_decode = MagicMock()
tokenizer.batch_decode.return_value = ["I HATE YOU", "I LOVE YOU"]
tokenizer.pad_token = "<PAD>"


class MagicWrappedModel(WrappedModel):
    def __init__(self):
        self.model = MagicMock()
        self.model.device = "cpu"
        self.inference_type = InferenceType.GENERATION
        self.left_tokenizer = self.right_tokenizer = tokenizer
        self.eval_minibatch_size = 2
        self.accelerator = MagicMock()
        self.accelerator.prepare = lambda data: data
        self.accelerator.gather_for_metrics = lambda data: data
        self.generation_config = None

    def generate(self, **inputs) -> torch.LongTensor:
        out = torch.tensor([[0], [1]], dtype=torch.long)
        assert isinstance(out, torch.LongTensor)
        return out

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> PreTrainedTokenizerBase:
        return tokenizer


def test_toxicity_in_generation_from_text_callback():
    victim = MagicWrappedModel()
    callback = build_binary_scoring_callback(
        AutoregressiveCallbackConfig(
            callback_name="binary_univariate_fn_of_generation_from_text",
            callback_return_type="binary",
            scoring_fn="doesnt_contain_toxicity",
            scoring_fn_arity=1,
        )
    )
    callback_input = CallbackInput(["Question : why?", "Question : really why?"])
    out = callback(victim, callback_input)
    assert isinstance(out, BinaryCallbackOutput)
    assert out.successes == [False, True]
    assert "I HATE YOU" in out.info["generations"][0]
    assert "I LOVE YOU" in out.info["generations"][1]
