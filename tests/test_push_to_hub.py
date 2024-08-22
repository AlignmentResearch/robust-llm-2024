# mypy: disable-error-code="attr-defined"
# pyright: reportFunctionMemberAccess=false
# Above lines disable mypy and pyright checks for MagicMock attributes

from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizerBase

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import WrappedModel
from robust_llm.models.model_utils import InferenceType


class DummyModel(MagicMock):

    def modules(self):
        return [MagicMock()]

    def num_parameters(self):
        return 0


class DummyTokenizer(MagicMock):
    push_to_hub = MagicMock()


class DummyAccelerator(MagicMock):
    device = "cpu"

    def prepare(self, model):
        return model


class DummyWrappedModel(WrappedModel):
    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> PreTrainedTokenizerBase:
        return MagicMock()

    def forward(self, *args, **kwargs):
        return MagicMock()


@pytest.fixture
def mock_model() -> DummyWrappedModel:
    model = DummyWrappedModel(
        model=DummyModel(),
        right_tokenizer=DummyTokenizer(),
        accelerator=DummyAccelerator(),
        inference_type=InferenceType.CLASSIFICATION,
        train_minibatch_size=2,
        eval_minibatch_size=2,
        family="dummy",
    )
    model.save_local = MagicMock()  # type: ignore[method-assign]
    model.model.push_to_hub = MagicMock()
    model.model._create_repo = MagicMock()
    model.model._upload_modified_files = MagicMock()
    return model


def test_push_to_hub_success(mock_model: DummyWrappedModel):
    mock_model.save_local.side_effect = None
    mock_model.model._create_repo.side_effect = lambda repo_id: repo_id
    mock_model.model._upload_modified_files.side_effect = None

    mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)
    mock_model.model._create_repo.assert_called_once_with("repo_id")


def test_push_to_hub_retry_success(mock_model: DummyWrappedModel):
    mock_model.save_local.side_effect = [Exception("Fail"), None]
    mock_model.model._create_repo.side_effect = lambda repo_id: repo_id
    mock_model.model._upload_modified_files.side_effect = None

    mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)
    assert mock_model.save_local.call_count == 2


def test_push_to_hub_failure(mock_model: DummyWrappedModel):
    mock_model.save_local.side_effect = Exception("Fail")
    mock_model.model._create_repo.side_effect = lambda repo_id: repo_id
    mock_model.model._upload_modified_files.side_effect = None

    with pytest.raises(RuntimeError):
        mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)


def test_push_to_hub_warnings(mock_model: DummyWrappedModel):
    mock_model.save_local.side_effect = [Exception("Fail"), None]
    mock_model.model._create_repo.side_effect = lambda repo_id: repo_id
    mock_model.model._upload_modified_files.side_effect = None

    with pytest.warns(UserWarning):
        mock_model.push_to_hub("repo_id", "revision", retries=3, cooldown_seconds=0.1)
