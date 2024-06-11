import hypothesis
import pytest
import torch
from accelerate import Accelerator
from hypothesis import example, given, settings
from hypothesis import strategies as st
from transformers import AutoTokenizer

from robust_llm.attacks.search_based.runners import GCGRunner, make_runner
from robust_llm.attacks.search_based.utils import (
    AttackIndices,
    PreppedExample,
    PromptTemplate,
    ReplacementCandidate,
    TokenizationChangeException,
)
from robust_llm.config import GCGAttackConfig
from robust_llm.models import GPT2Model, GPTNeoXModel
from robust_llm.models.model_utils import InferenceType
from robust_llm.utils import FakeModelForSequenceClassification

ACCELERATOR = Accelerator(cpu=True)


def gpt2_gcg_runner(before_attack_text: str, after_attack_text: str) -> GCGRunner:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    wrapped_model = GPT2Model(
        FakeModelForSequenceClassification(),  # type: ignore
        tokenizer,
        accelerator=ACCELERATOR,
        inference_type=InferenceType("classification"),
        train_minibatch_size=2,
        eval_minibatch_size=2,
        generation_config=None,
    )
    config = GCGAttackConfig(
        n_candidates_per_it=1,
        n_its=1,
        n_attack_tokens=11,
        top_k=1,
    )
    prompt_template = PromptTemplate(
        before_attack=before_attack_text, after_attack=after_attack_text
    )
    prepped_examples = [
        PreppedExample(
            prompt_template=prompt_template,
            clf_label=0,
            gen_target="^",
        )
    ]
    runner = make_runner(
        victim=wrapped_model,
        prepped_examples=prepped_examples,
        random_seed=0,
        config=config,
    )
    assert isinstance(runner, GCGRunner)
    return runner


def pythia_gcg_runner(before_attack_text: str, after_attack_text: str) -> GCGRunner:
    # we need a model for pythia because we access the config
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token

    wrapped_model = GPTNeoXModel(
        FakeModelForSequenceClassification(),  # type: ignore
        tokenizer,
        accelerator=ACCELERATOR,
        inference_type=InferenceType("classification"),
        train_minibatch_size=2,
        eval_minibatch_size=2,
        generation_config=None,
    )
    config = GCGAttackConfig(
        n_candidates_per_it=1,
        n_its=1,
        n_attack_tokens=11,
        top_k=1,
    )
    prompt_template = PromptTemplate(
        before_attack=before_attack_text, after_attack=after_attack_text
    )
    prepped_examples = [
        PreppedExample(
            prompt_template=prompt_template,
            clf_label=0,
            gen_target="^",
        )
    ]
    runner = make_runner(
        victim=wrapped_model,
        prepped_examples=prepped_examples,
        random_seed=0,
        config=config,
    )
    assert isinstance(runner, GCGRunner)
    return runner


RUNNERS = {
    "gpt2": gpt2_gcg_runner,
    "pythia": pythia_gcg_runner,
}


def get_gcg_runner(
    model_name: str,
    before_attack_text: str = "some before text",
    after_attack_text: str = "some after text",
) -> GCGRunner:
    return RUNNERS[model_name](before_attack_text, after_attack_text)


@pytest.fixture(params=RUNNERS.keys())
def gcg_runner(request) -> GCGRunner:
    return get_gcg_runner(request.param)


def test_AttackIndices() -> None:
    tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    attack = [2, 3, 4]
    target = [5, 6, 7, 8]
    attack_indices = AttackIndices(
        attack_start=2, attack_end=5, target_start=5, target_end=9
    )
    assert attack_indices.attack_start == 2
    assert attack_indices.attack_end == 5
    assert attack_indices.target_start == 5
    assert attack_indices.target_end == 9

    assert attack_indices.attack_length == 3
    assert attack_indices.target_length == 4

    assert tokens[attack_indices.attack_slice] == attack
    assert tokens[attack_indices.target_slice] == target


# hypothesis was generating hex characters that are not reversible in the tokenizer.
text_no_specials = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126))


@given(target=text_no_specials)
@example(target="Hi!")
@example(target="")
# We use a deadline of 1000ms because the default of 200ms was too short, as was
# a deadline of 500ms.
@settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=1000,
)
def test_get_attack_indices(gcg_runner: GCGRunner, target: str) -> None:
    # set up gcg runner to use the given user prompt and target
    gcg_runner.example.gen_target = target
    initial_attack_text = gcg_runner.initial_attack_text

    # compute token lengths for checking that exceptions are raised when necessary
    full_prompt = gcg_runner.example.prompt_template.build_prompt(
        attack_text=initial_attack_text, target=target
    )
    full_tokens = gcg_runner._get_tokens(full_prompt, return_tensors="pt")

    before_attack_tokens = gcg_runner._get_tokens(
        gcg_runner.example.prompt_template.before_attack, return_tensors="pt"
    )
    attack_tokens = gcg_runner._get_tokens(initial_attack_text, return_tensors="pt")
    after_attack_tokens = gcg_runner._get_tokens(
        gcg_runner.example.prompt_template.after_attack, return_tensors="pt"
    )
    target_tokens = gcg_runner._get_tokens(target, return_tensors="pt")

    concat_tokens = torch.cat(
        [before_attack_tokens, attack_tokens, after_attack_tokens, target_tokens],
        dim=1,
    )

    # run the actual tests
    attack_indices = None
    if not torch.equal(concat_tokens, full_tokens):
        with pytest.raises(TokenizationChangeException):
            attack_indices = gcg_runner._get_attack_indices(
                initial_attack_text, gcg_runner.example
            )
        return

    attack_indices = gcg_runner._get_attack_indices(
        initial_attack_text, gcg_runner.example
    )
    assert attack_indices is not None
    assert attack_indices.attack_length == gcg_runner.n_attack_tokens
    assert (
        attack_indices.target_length
        == gcg_runner._get_tokens(target, return_tensors="pt").shape[1]
    )

    new_attack_text = gcg_runner._decode_tokens(
        full_tokens[:, attack_indices.attack_slice]
    )
    assert new_attack_text == initial_attack_text

    new_target = gcg_runner._decode_tokens(full_tokens[:, attack_indices.target_slice])
    assert new_target == target


@pytest.mark.parametrize("model_name", RUNNERS.keys())
@pytest.mark.parametrize("before_attack_text", ["hi!@"])
def test_filter_candidates(model_name: str, before_attack_text: str) -> None:
    # TODO(GH#120): work out how to be consistent around whitespace
    # and then it'll be easier to write tests that work on all models

    gcg_runner = get_gcg_runner(model_name, before_attack_text)

    def get_token_id(s: str) -> int:
        tokens = gcg_runner._get_tokens(s, return_tensors="pt")
        assert tokens.shape == (1, 1)
        return tokens.squeeze().item()  # type: ignore

    # most tokenizers will merge adjacent '@'s so we use those for testing
    merge_char = "@"
    # need to update the attack indices if we change the user prompt
    gcg_runner.attack_indices = gcg_runner._get_attack_indices(
        gcg_runner.initial_attack_text,
        gcg_runner.example,
    )
    n_attack_tokens = gcg_runner.n_attack_tokens

    merge_char_token_id = get_token_id(merge_char)
    merge_with_user_prompt_candidate = ReplacementCandidate(0, merge_char_token_id)
    merge_inside_suffix_candidate = ReplacementCandidate(2, merge_char_token_id)
    merge_with_target_candidate = ReplacementCandidate(
        n_attack_tokens - 1, merge_char_token_id
    )
    # attack_tokens[1] should already be @ so this shouldn't change the suffix
    doesnt_change_candidate = ReplacementCandidate(1, merge_char_token_id)
    bad_candidates = [
        merge_with_user_prompt_candidate,
        merge_inside_suffix_candidate,
        merge_with_target_candidate,
        doesnt_change_candidate,
    ]
    attack_text = gcg_runner.initial_attack_text
    text_bad_replacement_pairs = [
        (attack_text, candidate) for candidate in bad_candidates
    ]
    bad_filtered_candidates = gcg_runner._filter_candidates(text_bad_replacement_pairs)
    assert len(bad_filtered_candidates) == 0

    # this char is sufficiently different that it shouldn't merge with @ or q
    nonmerge_char = "%"
    nonmerge_char_token_id = get_token_id(nonmerge_char)
    good_candidate_1 = ReplacementCandidate(0, nonmerge_char_token_id)
    good_candidate_2 = ReplacementCandidate(1, nonmerge_char_token_id)
    good_candidates = [good_candidate_1, good_candidate_2]
    text_good_replacement_pairs = [
        (attack_text, candidate) for candidate in good_candidates
    ]
    good_filtered_candidates = gcg_runner._filter_candidates(
        text_good_replacement_pairs
    )
    assert len(good_filtered_candidates) == 2


def test_get_replacement_candidates_from_gradients(gcg_runner: GCGRunner) -> None:
    gcg_runner.top_k = 2
    gcg_runner.n_candidates_per_it = 4

    vocab_size = gcg_runner.victim.right_tokenizer.vocab_size  # type: ignore
    # mocking gradients:
    # need to make it the same shape as it really would be because
    # some tokenizers (e.g. bert) care about special indices
    gradients = torch.zeros((2, vocab_size))
    # Don't use 0 as a token id because for some tokenizers it's a
    gradients[0, 100] = -1.0
    gradients[0, 102] = -1.0
    gradients[1, 101] = -1.0
    gradients[1, 103] = -1.0

    expected_candidates = set(
        [
            ReplacementCandidate(0, 100),
            ReplacementCandidate(0, 102),
            ReplacementCandidate(1, 101),
            ReplacementCandidate(1, 103),
        ]
    )
    actual_candidates = set(
        gcg_runner._get_replacement_candidates_from_gradients(gradients)
    )
    assert actual_candidates == expected_candidates


def test_replacements(gcg_runner: GCGRunner) -> None:
    initial_attack_text = "a!a!a"

    def get_token_id(s: str) -> int:
        tokens = gcg_runner._get_tokens(s, return_tensors="pt")
        assert tokens.shape == (1, 1)
        return tokens.squeeze().item()  # type: ignore

    replacement_candidates = [
        (ReplacementCandidate(0, get_token_id("b")), "b!a!a"),
        (ReplacementCandidate(1, get_token_id("c")), "aca!a"),
        (ReplacementCandidate(2, get_token_id("d")), "a!d!a"),
    ]

    for candidate, expected in replacement_candidates:
        updated = gcg_runner._decode_tokens(
            candidate.compute_tokens_after_replacement(
                gcg_runner._get_tokens(initial_attack_text, return_tensors="pt")
            )
        )
        assert updated == expected


def test_apply_replacements_and_eval_candidates(gcg_runner: GCGRunner) -> None:
    def get_token_id(s: str) -> int:
        tokens = gcg_runner._get_tokens(s, return_tensors="pt")
        assert tokens.shape == (1, 1)
        return tokens.squeeze().item()  # type: ignore

    initial_attack_text = "@!@!@!"
    gcg_runner.n_attack_tokens = gcg_runner._get_tokens(
        initial_attack_text, return_tensors="pt"
    ).shape[1]
    text_replacement_pairs = [
        (initial_attack_text, ReplacementCandidate(0, get_token_id("b"))),  # b!@!@!
        (initial_attack_text, ReplacementCandidate(1, get_token_id("c"))),  # @c@!@!
    ]
    expected_final_texts = ["b!@!@!", "@c@!@!"]

    scores_and_final_texts = gcg_runner._apply_replacements_and_eval_candidates(
        text_replacement_pairs
    )
    assert expected_final_texts == [t for _, t in scores_and_final_texts]


def test_chat_prompt_template():
    before_attack = "You are a helpful assistant.\nUser: Hello."
    after_attack = "\n\nAssistant: "

    target = "Hi, I'm an assistant."

    prompt_template = PromptTemplate(before_attack, after_attack)
    actual = prompt_template.build_prompt(target=target)

    expected = """
You are a helpful assistant.
User: Hello.

Assistant: Hi, I'm an assistant.""".strip()

    assert actual == expected


@given(st.text(), st.text(), st.text(), st.text())
def test_prompt_template(before_attack, attack_text, after_attack, target):
    prompt_template = PromptTemplate(before_attack, after_attack)
    actual = prompt_template.build_prompt(attack_text=attack_text, target=target)
    expected = before_attack + attack_text + after_attack + target
    assert actual == expected


def test_n_best_candidates_to_keep(gcg_runner: GCGRunner) -> None:
    assert gcg_runner.n_best_candidates_to_keep == 1
