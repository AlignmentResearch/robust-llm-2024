from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from robust_llm.attacks.random_token import RandomTokenAttack
from robust_llm.configs import OverallConfig
from robust_llm.pipelines.utils import prepare_victim_models

# Get an overall config and change the random token attack
# max and min to be the same to avoid randomness in tests
overall_config = OverallConfig()
attack_config = overall_config.experiment.evaluation.evaluation_attack

# Get victim model and tokenizer
victim_model, victim_tokenizer, _ = prepare_victim_models(overall_config)

# Set up the attack
attack = RandomTokenAttack(
    attack_config=attack_config,
    environment_config=overall_config.experiment.environment,
    modifiable_chunks_spec=[True, False, True],  # type: ignore
    dataset_type="tensor_trust",
    victim_model=victim_model,
    victim_tokenizer=victim_tokenizer,
    ground_truth_label_fn=None,
)


def _get_new_chunks_and_spec(
    chunks: Sequence[str], attack: RandomTokenAttack
) -> Tuple[List[str], List[bool | str]]:
    assert attack.attack_config.append_to_modifiable_chunk is True

    spec: List[bool | str] = []
    updated_chunks = []
    for chunk, s in zip(chunks, attack.modifiable_chunks_spec):
        updated_chunks.append(chunk)
        spec.append(s)
        if s is True:
            updated_chunks.append("new_chunk")
            spec.append("new")

    return updated_chunks, spec


def _sequential_get_adversarial_tokens(
    rt_attack: RandomTokenAttack, chunked_datapoint: Sequence[str]
) -> List[str]:
    """Get new random tokens in the modifiable chunks.

    Operates on a single chunked datapoint (a string which has been split into
    "chunks", some of which are modifiable, some of which are not, as determined
    by the self.modifiable_chunks_spec). This method replaces those chunks which
    are modifiable with random tokens from the victim tokenizer's vocabulary.
    If attack_config.append_to_modifiable_chunk is True, instead of replacing
    the modifiable chunks, new chunks are made and appended to them.

    Args:
        rt_attack: The random token attack object
        chunked_datapoint: The datapoint to operate on

    Returns:
        The chunked datapoint, with the modifiable chunks replaced
        with random tokens (or with new chunks appearing immediately
        after them).
    """
    new_chunked_datapoint: List[str] = []

    for chunk, is_modifiable in zip(
        chunked_datapoint, rt_attack.modifiable_chunks_spec
    ):
        if not is_modifiable:
            new_chunked_datapoint.append(chunk)
        else:
            num_tokens = int(
                torch.randint(
                    low=rt_attack.min_tokens,
                    high=rt_attack.max_tokens + 1,
                    size=(),
                    generator=rt_attack.torch_rng,
                ).item()
            )
            random_tokens = torch.randint(
                low=0,
                high=rt_attack.victim_tokenizer.vocab_size,  # type: ignore
                size=(num_tokens,),
                generator=rt_attack.torch_rng,
            )
            random_token_text = rt_attack.victim_tokenizer.decode(random_tokens)

            if rt_attack.attack_config.append_to_modifiable_chunk:
                new_chunked_datapoint.append(chunk)
            new_chunked_datapoint.append(random_token_text)

    return new_chunked_datapoint


def _sequential_check_success(
    rt_attack: RandomTokenAttack,
    attacked_chunked_datapoint: Sequence[str],
    original_label: int,
) -> bool:
    """Check whether the attack was successful on a single datapoint.

    This is done by running the victim model on the attacked text and comparing
    the result with the ground truth label. If the attack does not have a ground
    truth label function, the original label is used instead.

    Args:
        rt_attack: The random token attack object
        attacked_chunked_datapoint: The attacked text
        original_label: The original label of the text. Will only be
            used if the attack does not have a ground truth label function.

    Returns:
        True if the attack was successful, False otherwise.
    """

    convo = "".join(attacked_chunked_datapoint)
    result = rt_attack.victim_pipeline(convo)

    result_label = result[0]["label"]  # type: ignore
    result_int_label = rt_attack.victim_model.config.label2id[result_label]  # type: ignore # noqa: E501

    if rt_attack.ground_truth_label_fn is not None:
        true_label = rt_attack.ground_truth_label_fn(convo)
    else:
        true_label = original_label

    return result_int_label != true_label


# Override the pipeline with something simple
def custom_pipeline(
    input: str | List[str], batch_size: Optional[int] = None
) -> List[Dict[str, Any]]:

    def check(input: str) -> bool:
        return input in {"ABC", "DEF", "GHI", "JKL"}

    if isinstance(input, str):
        input = [input]

    labels = [check(i) for i in input]
    return [{"label": f"LABEL_{int(x)}"} for x in labels]


attack.victim_pipeline = custom_pipeline  # type: ignore


def _test_get_adversarial_tokens(rt_attack: RandomTokenAttack):
    """This test checks batched and sequential implementations of check_success.

    Specifically, it compares them against each other, and against handwritten values.
    The sequential implementation is written directly in this module.
    """

    text_chunked = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"], ["J", "K", "L"]]
    successes = [True, False, False, True]

    sequential_adversarial_tokens = []
    for chunks, success in zip(text_chunked, successes):
        if success:
            sequential_adversarial_tokens.append(chunks)
        else:
            adversarial_tokens = _sequential_get_adversarial_tokens(
                rt_attack=attack, chunked_datapoint=chunks
            )
            sequential_adversarial_tokens.append(adversarial_tokens)

    batched_adversarial_tokens = attack._batch_get_adversarial_tokens(
        chunked_datapoints=text_chunked, successes=successes
    )

    assert len(sequential_adversarial_tokens) == len(batched_adversarial_tokens)
    for seq, bat in zip(sequential_adversarial_tokens, batched_adversarial_tokens):
        assert len(seq) == len(bat)

    for seq, bat, orig, suc in zip(
        sequential_adversarial_tokens,
        batched_adversarial_tokens,
        text_chunked,
        successes,
    ):
        if suc:
            assert seq == bat == orig
        else:
            if rt_attack.attack_config.append_to_modifiable_chunk:
                assert (
                    len(seq)
                    == len(bat)
                    == len(orig) + sum(attack.modifiable_chunks_spec)
                )

                new_chunks, new_spec = _get_new_chunks_and_spec(orig, attack)

                for s, b, o, mod in zip(seq, bat, new_chunks, new_spec):
                    if mod == "new":
                        assert s != b != o
                    else:
                        assert s == b == o

            else:
                for s, b, o, mod in zip(seq, bat, orig, attack.modifiable_chunks_spec):
                    if mod:
                        # Probability of collision is low enough
                        # we don't need to care
                        assert s != b != o
                    else:
                        assert s == b == o


def test_get_adversarial_tokens_replace():
    previous_value = attack.attack_config.append_to_modifiable_chunk
    attack.attack_config.append_to_modifiable_chunk = False
    _test_get_adversarial_tokens(attack)
    attack.attack_config.append_to_modifiable_chunk = previous_value


def test_get_adversarial_tokens_append():
    previous_value = attack.attack_config.append_to_modifiable_chunk
    attack.attack_config.append_to_modifiable_chunk = True
    _test_get_adversarial_tokens(attack)
    attack.attack_config.append_to_modifiable_chunk = previous_value


def test_check_success():
    """This test checks batched and sequential implementations of check_success.

    Specifically, it compares them against each other, and against handwritten values.
    The sequential implementation is written directly in this module.
    """

    attacked_text_chunked = [
        ["C", "B", "C"],
        ["D", "E", "F"],
        ["G", "H", "X"],
        ["J", "Z", "L"],
    ]
    original_labels = [1, 1, 1, 1]
    previous_successes = [False, False, True, False]
    expected_successes = [1, 0, 1, 1]

    sequential_successes = []
    for chunk, label in zip(attacked_text_chunked, original_labels):
        success = _sequential_check_success(
            rt_attack=attack, attacked_chunked_datapoint=chunk, original_label=label
        )
        sequential_successes.append(success)

    batched_successes = attack._batch_check_success(
        attacked_chunked_datapoints=attacked_text_chunked,
        original_labels=original_labels,
        previous_attack_success=previous_successes,
    )

    assert (
        len(sequential_successes) == len(batched_successes) == len(expected_successes)
    )
    for seq, bat, exp in zip(
        sequential_successes, batched_successes, expected_successes
    ):
        assert seq == bat == exp
