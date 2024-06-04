import torch

from robust_llm.models.model_utils import generation_successes_from_logits


def test_generation_successes_from_logits():
    logits = torch.tensor(
        [
            [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.0, 0.0], [0.0, 0.0]],
            [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )
    goal = [[1, 0, 1], [0, 0, 1]]

    out = generation_successes_from_logits(logits, attention_mask, goal)
    # The first example succeeds because we can see that the non-padded tokens
    # predict [1, 0, 1, 1]. Predictions made FROM the goal tokens take up the
    # last three positions. Therefore predictions OF the goal tokens are shifted
    # over one to the left. Thus if the goal is [1, 0, 1] then it's correct,
    # but any other 3-token goal is incorrect.
    assert out == [True, False]
