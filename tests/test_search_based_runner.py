from unittest.mock import MagicMock

from robust_llm.attacks.search_based.runners.search_based_runner import (
    SearchBasedRunner,
)


def test_select_next_candidates():
    mock_runner = MagicMock()
    mock_runner.n_best_candidates_to_keep = 1
    fn = SearchBasedRunner._select_next_candidates
    cands, indices = fn(mock_runner, [(0.2, "a"), (0.1, "b"), (0.3, "c"), (0.4, "d")])
    assert cands == ["b"]
    assert indices == [1]
