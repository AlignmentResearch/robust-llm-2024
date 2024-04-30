from typing import Any, Optional, Sequence

import torch

from robust_llm.attacks.random_token import RandomTokenAttack
from robust_llm.config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    RandomTokenAttackConfig,
)
from robust_llm.pipelines.utils import prepare_victim_models
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset

# Get an overall config and change the random token attack
# max and min to be the same to avoid randomness in tests
exp_config = ExperimentConfig(
    experiment_type="evaluation",
    dataset=DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        n_val=10,
    ),
    model=ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
    ),
    evaluation=EvaluationConfig(
        evaluation_attack=RandomTokenAttackConfig(
            min_tokens=3,
            max_tokens=3,
        )
    ),
)
assert exp_config.evaluation is not None
attack_config = exp_config.evaluation.evaluation_attack
dataset_config = exp_config.dataset
dataset = load_rllm_dataset(dataset_config, split="validation")
# hack to get around the fact that the dataset DOES have a ground_truth_label_fn
# but it's not used in the test
dataset.ground_truth_label_fn = lambda text, label: label  # type: ignore[method-assign]

# Get victim model and tokenizer
victim_model, victim_tokenizer, _ = prepare_victim_models(exp_config, num_classes=2)

# Set up the attack
assert isinstance(attack_config, RandomTokenAttackConfig)
attack = RandomTokenAttack(
    attack_config=attack_config,
    victim_model=victim_model,
    victim_tokenizer=victim_tokenizer,
)


def _sequential_get_adversarial_tokens(
    rt_attack: RandomTokenAttack, chunked_datapoint: Sequence[str], dataset: RLLMDataset
) -> list[str]:
    """Get new random tokens in the modifiable chunks.

    Operates on a single chunked datapoint (a string which has been split into
    "chunks", some of which are modifiable, some of which are not, as determined
    by dataset.modifiable_chunk_spec). This method replaces those chunks which
    are modifiable with random tokens from the victim tokenizer's vocabulary.
    If the ChunkType is PERTURBABLE, instead of replacing
    the modifiable chunk, a new chunk is made and appended to it.

    Args:
        rt_attack: The random token attack object
        chunked_datapoint: The datapoint to operate on

    Returns:
        The chunked datapoint, with the modifiable chunks replaced
        with random tokens (or with new chunks appearing immediately
        after them).
    """
    new_chunked_datapoint: list[str] = []

    for chunk, chunk_type in zip(chunked_datapoint, dataset.modifiable_chunk_spec):
        if chunk_type == ChunkType.IMMUTABLE:
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

            if chunk_type == ChunkType.PERTURBABLE:
                new_chunked_datapoint.append(chunk)
            new_chunked_datapoint.append(random_token_text)

    return new_chunked_datapoint


def _sequential_check_success(
    rt_attack: RandomTokenAttack,
    attacked_chunked_datapoint: Sequence[str],
    original_label: int,
    dataset: RLLMDataset,
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

    true_label = dataset.ground_truth_label_fn(convo, original_label)

    return result_int_label != true_label


# Override the pipeline with something simple
def custom_pipeline(
    input: str | list[str], batch_size: Optional[int] = None
) -> list[dict[str, Any]]:

    def check(input: str) -> bool:
        return input in {"ABC", "DEF", "GHI", "JKL"}

    if isinstance(input, str):
        input = [input]

    labels = [check(i) for i in input]
    return [{"label": f"LABEL_{int(x)}"} for x in labels]


attack.victim_pipeline = custom_pipeline  # type: ignore


def _test_get_adversarial_tokens(rt_attack: RandomTokenAttack, dataset: RLLMDataset):
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
                rt_attack=attack, chunked_datapoint=chunks, dataset=dataset
            )
            sequential_adversarial_tokens.append(adversarial_tokens)

    batched_adversarial_tokens = attack._batch_get_adversarial_tokens(
        chunked_datapoints=text_chunked, successes=successes, dataset=dataset
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


def test_get_adversarial_tokens():
    _test_get_adversarial_tokens(attack, dataset)


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
            rt_attack=attack,
            attacked_chunked_datapoint=chunk,
            original_label=label,
            dataset=dataset,
        )
        sequential_successes.append(success)

    batched_successes = attack._batch_check_success(
        attacked_chunked_datapoints=attacked_text_chunked,
        original_labels=original_labels,
        previous_attack_success=previous_successes,
        dataset=dataset,
    )

    assert (
        len(sequential_successes) == len(batched_successes) == len(expected_successes)
    )
    for seq, bat, exp in zip(
        sequential_successes, batched_successes, expected_successes
    ):
        assert seq == bat == exp
