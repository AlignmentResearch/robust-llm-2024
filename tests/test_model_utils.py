import torch

from robust_llm.models.model_utils import generation_successes_from_logits


def test_generation_successes_from_logits():
    input_ids = torch.tensor(
        [
            # 100 is first token (which is not predicted), 101 is pad
            [100, 0, 1, 0, 1, 101],
            [100, 0, 1, 0, 1, 101],
            [100, 0, 1, 0, 1, 101],
        ]
    ).tolist()

    logits = torch.tensor(
        [
            # Argmax of the first example is [1, 1, 0, 1, 1, ?]
            # Zeroth logit is based on input_ids[0]=100 and predicts input_ids[1]=0
            # First logit is based on input_ids[1]=0 and predicts input_ids[2]=1
            # Second logit is based on input_ids[2]=1 and predicts input_ids[3]=0
            # Third logit is based on input_ids[3]=0 and predicts input_ids[4]=1
            # Fourth logit is based on input_ids[4]=1 and predicts input_ids[5]=101
            [[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.0, 0.0]],
            # Argmax of the second example is [1, 0, 0, 1, 1, ?]
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.0, 0.0]],
            # Argmax of the third example is [1, 0, 1, 0, 1, ?]
            [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.0, 0.0]],
        ]
    )
    goal = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]

    out = generation_successes_from_logits(
        logits=logits, input_ids=input_ids, goal=goal
    )
    # The first example succeeds because we can see that the predicted first
    # three tokens (which are the ones that line up with the goal in the
    # input_ids) are [1, 0, 1].
    # The second example fails because the predicted first three tokens are
    # [1, 0, 0].
    # Third example fails because [1, 0, 1] appears twice but not in the right
    # place.
    # Predictions made FROM the goal tokens take up the last three positions.
    # Therefore predictions OF the goal tokens are shifted over one to the left.
    # Thus if the goal is [1, 0, 1] then it's correct, but any other 3-token
    # goal is incorrect.
    assert out == [True, False, False]


def test_generation_successes_from_logits_with_caching():
    """With caching we get fewer logits back."""
    input_ids = torch.tensor(
        [
            # 100 is first token (which is not predicted), 101 is pad
            [100, 0, 1, 0, 1, 101],
            [100, 0, 1, 0, 1, 101],
        ]
    ).tolist()

    logits = torch.tensor(
        [
            # Argmax of the first example is [1, 1, 0, 1, 1, ?]
            # Zeroth logit is based on input_ids[0]=100 and predicts input_ids[1]=0
            # First logit is based on input_ids[1]=0 and predicts input_ids[2]=1
            # Second logit is based on input_ids[2]=1 and predicts input_ids[3]=0
            # Third logit is based on input_ids[3]=0 and predicts input_ids[4]=1
            # Fourth logit is based on input_ids[4]=1 and predicts input_ids[5]=101
            [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.0, 0.0]],
            # Argmax of the second example is [1, 0, 0, 1, 1, ?]
            [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.0, 0.0]],
        ]
    )
    goal = [[1, 0, 1], [1, 0, 1]]

    out = generation_successes_from_logits(
        logits=logits, input_ids=input_ids, goal=goal
    )
    # The first example succeeds because we can see that the predicted first
    # three tokens (which are the ones that line up with the goal in the
    # input_ids) are [1, 0, 1]. The second example fails because the predicted
    # first three tokens are [1, 0, 0].
    # predict [1, 0, 1, 1]. Predictions made FROM the goal tokens take up the
    # last three positions. Therefore predictions OF the goal tokens are shifted
    # over one to the left. Thus if the goal is [1, 0, 1] then it's correct,
    # but any other 3-token goal is incorrect.
    assert out == [True, False]
