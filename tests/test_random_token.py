"""Tests for random token attack.

Methods to test:
    "RandomTokenAttack",
    "get_randint_with_exclusions",
"""

import dataclasses
from unittest import mock
from unittest.mock import MagicMock, PropertyMock

import pytest
from transformers import AutoTokenizer

from robust_llm.attacks.search_free.random_token import RandomTokenAttack
from robust_llm.attacks.search_free.search_free import get_first_attack_success_index
from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.config.configs import EvaluationConfig, ExperimentConfig
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)


@pytest.fixture
def random_token_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_type="evaluation",
        evaluation=EvaluationConfig(
            evaluation_attack=RandomTokenAttackConfig(
                n_attack_tokens=3,
                save_total_limit=0,
            )
        ),
    )


@pytest.fixture
def mocked_victim():
    victim = MagicMock()
    victim.decode = lambda x: victim.right_tokenizer.decode(x)
    victim.batch_decode = lambda x: victim.right_tokenizer.batch_decode(x)
    victim.vocab_size = 10
    return victim


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


def test_build_random_token_attack(random_token_config):
    attack = RandomTokenAttack(random_token_config, MagicMock(), False)
    assert attack.n_attack_tokens == 3


def test_n_random_token_ids_with_exclusions(random_token_config, mocked_victim):
    n_tokens = 100
    attack = RandomTokenAttack(random_token_config, mocked_victim, False)
    exclusions = [0, 1, 2, 3, 4, 5]
    tokens = attack._n_random_token_ids_with_exclusions(n_tokens, exclusions)
    assert len(tokens) == n_tokens
    assert all([x not in exclusions for x in tokens])
    assert all([x < 10 for x in tokens])
    assert all([x >= 0 for x in tokens])


def test_first_success_index(random_token_config):

    successes = [True, False, True]
    rv = get_first_attack_success_index(successes)
    assert rv == 1

    successes = [True, True, True]
    rv = get_first_attack_success_index(successes)
    assert rv == 2

    successes = [False, False, False]
    rv = get_first_attack_success_index(successes)
    assert rv == 0


def test_get_text_for_chunk(random_token_config, mocked_victim, tokenizer):
    excluded_token_ids = [0, 1, 2, 3, 4, 5]
    mocked_victim.right_tokenizer = tokenizer
    # Mock the all_special_ids property of the tokenizer
    with mock.patch(
        "transformers.GPT2TokenizerFast.all_special_ids", new_callable=PropertyMock
    ) as mock_specials:
        mock_specials.return_value = excluded_token_ids
        assert random_token_config.evaluation is not None
        attack_config = dataclasses.replace(
            random_token_config.evaluation.evaluation_attack, n_attack_tokens=10
        )
        config = dataclasses.replace(
            random_token_config,
            evaluation=EvaluationConfig(evaluation_attack=attack_config),
        )
        attack = RandomTokenAttack(config, mocked_victim, False)

    chunk_text = "Chunk text"

    chunk_type = ChunkType.IMMUTABLE
    rv, _ = attack._get_text_for_chunk(
        chunk_text,
        chunk_type,
        current_iteration=0,
        chunk_proxy_label=1,
        chunk_seed=0,
        chunk_index=0,
    )
    assert rv == "Chunk text"

    chunk_type = ChunkType.PERTURBABLE
    rv, _ = attack._get_text_for_chunk(
        chunk_text,
        chunk_type,
        current_iteration=0,
        chunk_proxy_label=1,
        chunk_seed=1,
        chunk_index=1,
    )
    assert rv.startswith("Chunk text")
    assert rv != "Chunk text"

    chunk_type = ChunkType.OVERWRITABLE
    rv, _ = attack._get_text_for_chunk(
        chunk_text,
        chunk_type,
        current_iteration=0,
        chunk_proxy_label=1,
        chunk_seed=2,
        chunk_index=2,
    )
    assert not rv.startswith("Chunk text")


def test_get_attacked_input(random_token_config, mocked_victim, tokenizer):
    """Test the _get_attacked_input method of the RandomTokenAttack class.

    TODO (ian): Write tests with a custom tokenizer?
    """
    mocked_victim.right_tokenizer = tokenizer
    attack = RandomTokenAttack(random_token_config, mocked_victim, False)
    chunked_datapoint = ["a", "b", "c"]
    example = {
        "chunked_text": chunked_datapoint,
        "text": "".join(chunked_datapoint),
        "clf_label": 1,
        "proxy_clf_label": 0,
        "gen_target": "POSITIVE",
        "proxy_gen_target": "NEGATIVE",
        "seed": 0,
    }
    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.IMMUTABLE, ChunkType.IMMUTABLE
    )
    attacked_input, _ = attack._get_attacked_input(
        example, modifiable_chunk_spec, current_iteration=0
    )
    assert attacked_input == "abc"

    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.IMMUTABLE, ChunkType.PERTURBABLE
    )
    attacked_input, _ = attack._get_attacked_input(
        example, modifiable_chunk_spec, current_iteration=0
    )
    assert attacked_input.startswith("abc")
    assert attacked_input != "abc"

    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.OVERWRITABLE, ChunkType.IMMUTABLE, ChunkType.IMMUTABLE
    )
    attacked_input, _ = attack._get_attacked_input(
        example, modifiable_chunk_spec, current_iteration=0
    )
    assert attacked_input.endswith("c")
    # The two characters from the original chunks, plus at least 1 random tokens
    # from the modifiable chunk (In practice it'll be much more than
    # this, this is just a lower bound.)
    assert len(attacked_input) >= 5
