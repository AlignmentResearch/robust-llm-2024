from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizerBase

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import WrappedModel
from robust_llm.models.model_utils import InferenceType


class DummyModel:

    push_to_hub = MagicMock()


class DummyTokenizer:

    push_to_hub = MagicMock()


class DummyWrappedModel(WrappedModel):
    model = DummyModel()
    tokenizer = DummyTokenizer()

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> PreTrainedTokenizerBase:
        return MagicMock()


@pytest.fixture
def mock_model() -> DummyWrappedModel:
    return DummyWrappedModel(
        model=MagicMock(),
        tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.CLASSIFICATION,
        train_minibatch_size=2,
        eval_minibatch_size=2,
    )


def test_push_to_hub_success(mock_model: DummyWrappedModel):
    mock_model.model.push_to_hub.return_value = None
    mock_model.tokenizer.push_to_hub.return_value = None
    mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)
    mock_model.model.push_to_hub.assert_called_once_with(
        repo_id="repo_id", revision="revision"
    )


def test_push_to_hub_retry_success(mock_model: DummyWrappedModel):
    mock_model.model.push_to_hub.side_effect = [Exception("Fail"), None]
    mock_model.tokenizer.push_to_hub.side_effect = None
    mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)
    assert mock_model.model.push_to_hub.call_count == 2
    assert mock_model.tokenizer.push_to_hub.call_count == 1


def test_push_to_hub_failure(mock_model: DummyWrappedModel):
    mock_model.model.push_to_hub.side_effect = None
    mock_model.tokenizer.push_to_hub.side_effect = Exception("Fail")
    with pytest.raises(RuntimeError):
        mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)
    assert mock_model.model.push_to_hub.call_count == 3
    assert mock_model.tokenizer.push_to_hub.call_count == 3


def test_push_to_hub_warnings(mock_model: DummyWrappedModel):
    mock_model.model.push_to_hub.side_effect = [Exception("Fail"), None]
    mock_model.tokenizer.push_to_hub.side_effect = None
    with pytest.warns(UserWarning):
        mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)
