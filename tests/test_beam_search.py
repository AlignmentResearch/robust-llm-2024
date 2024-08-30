import hypothesis
import pytest
from accelerate import Accelerator
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import AutoTokenizer

from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.runners.beam_search_runner import BeamSearchRunner
from robust_llm.attacks.search_based.utils import PreppedExample, PromptTemplate
from robust_llm.config import BeamSearchAttackConfig
from robust_llm.mocks import FakeModelForSequenceClassification
from robust_llm.models import GPT2Model, GPTNeoXModel, WrappedModel
from robust_llm.models.model_utils import InferenceType


def _get_runner(
    before_attack_text: str,
    after_attack_text: str,
    beam_search_width: int,
    victim: WrappedModel,
) -> BeamSearchRunner:
    config = BeamSearchAttackConfig(beam_search_width=beam_search_width)
    prompt_template = PromptTemplate(
        before_attack=before_attack_text, after_attack=after_attack_text
    )
    prepped_examples = [
        PreppedExample(
            prompt_template=prompt_template,
            clf_label=0,
            gen_target="0",
        )
    ]
    runner = make_runner(
        victim=victim,
        prepped_examples=prepped_examples,
        random_seed=0,
        n_its=2,
        config=config,
    )
    assert isinstance(runner, BeamSearchRunner)
    return runner


MODEL_SPECS = {
    # (class, family, name)
    (GPT2Model, "gpt2", "gpt2"),
    (GPTNeoXModel, "pythia", "EleutherAI/pythia-70m-deduped"),
}
MODEL_KEYS = [family for _, family, _ in MODEL_SPECS]


# Though this could make sense as a global variable, we don't want to initialize
# Accelerator(use_cpu=True) globally because it can mess up GPU tests. Even if
# no tests in this file run, this file may get executed in pytest's collection
# phase and cause GPU tests to unexpectedly be run on the CPU.
@pytest.fixture(scope="module")
def wrapped_models() -> dict[str, WrappedModel]:
    accelerator = Accelerator(cpu=False)
    return {
        family: cls(
            FakeModelForSequenceClassification(),  # type: ignore
            AutoTokenizer.from_pretrained(name),
            accelerator=accelerator,
            inference_type=InferenceType("classification"),
            train_minibatch_size=2,
            eval_minibatch_size=2,
            generation_config=None,
            family=family,
        )
        for cls, family, name in MODEL_SPECS
    }


def get_beam_search_runner(
    wrapped_model: WrappedModel,
    before_attack_text: str = "some before text",
    after_attack_text: str = "some after text",
    beam_search_width: int = 5,
) -> BeamSearchRunner:
    return _get_runner(
        before_attack_text=before_attack_text,
        after_attack_text=after_attack_text,
        beam_search_width=beam_search_width,
        victim=wrapped_model,
    )


@pytest.fixture(params=MODEL_KEYS)
def beam_search_runner(request, wrapped_models) -> BeamSearchRunner:
    return get_beam_search_runner(wrapped_models[request.param])


@pytest.mark.parametrize("model_name", MODEL_KEYS)
@pytest.mark.parametrize("beam_search_width", [5, 13])
def test_n_best_candidates_to_keep(
    wrapped_models, model_name: str, beam_search_width: int
) -> None:
    runner = get_beam_search_runner(
        wrapped_models[model_name], beam_search_width=beam_search_width
    )
    assert runner.n_best_candidates_to_keep == beam_search_width


@given(initial_text=st.text())
@pytest.mark.parametrize("num_initial_candidates", [1, 5])
@settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=1000,
)
def test_get_candidate_texts_and_replacements(
    beam_search_runner: BeamSearchRunner,
    initial_text: str,
    num_initial_candidates: int,
) -> None:
    candidate_texts = [initial_text] * num_initial_candidates
    candidate_texts_and_replacements = (
        beam_search_runner._get_candidate_texts_and_replacements(candidate_texts)
    )
    assert (
        len(candidate_texts_and_replacements) == beam_search_runner.n_candidates_per_it
    )
