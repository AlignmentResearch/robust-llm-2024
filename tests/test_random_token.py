"""
Methods to test:
    "RandomTokenAttack",
    "get_randint_with_exclusions",
"""

from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from robust_llm.attacks.random_token import (
    RandomTokenAttack,
    get_attacked_text_from_successes,
)
from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)


@pytest.fixture
def random_token_config():
    return RandomTokenAttackConfig(n_attack_tokens=3, n_its=100)


@pytest.fixture
def mocked_victim():
    victim = MagicMock()
    victim.vocab_size = 10
    return victim


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


def test_build_random_token_attack(random_token_config):
    attack = RandomTokenAttack(random_token_config, MagicMock())
    assert attack.n_attack_tokens == 3
    assert attack.n_its == 100


def test_n_random_token_ids_with_exclusions(random_token_config, mocked_victim):
    n_tokens = 100
    attack = RandomTokenAttack(random_token_config, mocked_victim)
    exclusions = [0, 1, 2, 3, 4, 5]
    tokens = attack._n_random_token_ids_with_exclusions(n_tokens, exclusions)
    assert len(tokens) == n_tokens
    assert all([x not in exclusions for x in tokens])
    assert all([x < 10 for x in tokens])
    assert all([x >= 0 for x in tokens])


def test_get_attacked_text_from_successes():
    attacked_inputs = ["a", "b", "c"]

    successes = [True, False, True]
    rv = get_attacked_text_from_successes(attacked_inputs, successes)
    assert rv == ("b", 1)

    successes = [True, True, True]
    rv = get_attacked_text_from_successes(attacked_inputs, successes)
    assert rv == ("c", None)

    successes = [False, False, False]
    rv = get_attacked_text_from_successes(attacked_inputs, successes)
    assert rv == ("a", 0)


def test_get_text_for_chunk(random_token_config, mocked_victim, tokenizer):
    mocked_victim.tokenizer = tokenizer
    attack = RandomTokenAttack(random_token_config, mocked_victim)
    chunk_text = "Chunk text"
    n_tokens = 10
    excluded_tokens = [0, 1, 2, 3, 4, 5]

    chunk_type = ChunkType.IMMUTABLE
    rv = attack._get_text_for_chunk(chunk_text, chunk_type, excluded_tokens, n_tokens)
    assert rv == "Chunk text"

    chunk_type = ChunkType.PERTURBABLE
    rv = attack._get_text_for_chunk(chunk_text, chunk_type, excluded_tokens, n_tokens)
    assert rv.startswith("Chunk text")
    assert rv != "Chunk text"

    chunk_type = ChunkType.OVERWRITABLE
    rv = attack._get_text_for_chunk(chunk_text, chunk_type, excluded_tokens, n_tokens)
    assert not rv.startswith("Chunk text")


def test_get_attacked_input(random_token_config, mocked_victim, tokenizer):
    """Test the _get_attacked_input method of the RandomTokenAttack class.

    TODO (ian): Write tests with a custom tokenizer?
    """
    mocked_victim.tokenizer = tokenizer
    attack = RandomTokenAttack(random_token_config, mocked_victim)
    chunked_datapoint = ["a", "b", "c"]
    example = {
        "chunked_text": chunked_datapoint,
        "text": "".join(chunked_datapoint),
        "clf_label": 1,
        "gen_target": "POSITIVE",
    }
    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.IMMUTABLE, ChunkType.IMMUTABLE
    )
    attacked_input = attack._get_attacked_input(example, modifiable_chunk_spec)
    assert attacked_input == "abc"

    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.IMMUTABLE, ChunkType.PERTURBABLE
    )
    attacked_input = attack._get_attacked_input(example, modifiable_chunk_spec)
    assert attacked_input.startswith("abc")
    assert attacked_input != "abc"

    modifiable_chunk_spec = ModifiableChunkSpec(
        ChunkType.OVERWRITABLE, ChunkType.PERTURBABLE, ChunkType.IMMUTABLE
    )
    attacked_input = attack._get_attacked_input(example, modifiable_chunk_spec)
    assert attacked_input.endswith("c")
    # The three characters from the original chunks, plus at least 3 random
    # tokens from the first two modifiable chunks, minus 1 for the OVERWRITABLE
    # chunk. (In practice it'll be much more than this, this is just a lower
    # bound.)
    assert len(attacked_input) >= 8
