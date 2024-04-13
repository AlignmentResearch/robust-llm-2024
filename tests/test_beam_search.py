import hypothesis
import pytest
from accelerate import Accelerator
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import AutoTokenizer

from robust_llm.attacks.search_based.models import (
    SearchBasedAttackWrappedModel,
    WrappedBERTModel,
    WrappedGPT2Model,
    WrappedGPTNeoXModel,
)
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.runners.beam_search_runner import BeamSearchRunner
from robust_llm.attacks.search_based.utils import PreppedExample, PromptTemplate
from robust_llm.configs import BeamSearchAttackConfig, SearchBasedAttackConfig
from robust_llm.utils import FakeModelForSequenceClassification


def _get_runner(
    before_attack_text: str,
    after_attack_text: str,
    beam_search_width: int,
    wrapped_model: SearchBasedAttackWrappedModel,
) -> BeamSearchRunner:
    config = SearchBasedAttackConfig(
        search_type="beam_search",
        beam_search_attack_config=BeamSearchAttackConfig(
            beam_search_width=beam_search_width
        ),
    )
    prompt_template = PromptTemplate(
        before_attack=before_attack_text, after_attack=after_attack_text
    )
    prepped_examples = [
        PreppedExample(
            prompt_template=prompt_template,
            clf_target=0,
        )
    ]
    runner = make_runner(
        wrapped_model=wrapped_model,
        prepped_examples=prepped_examples,
        random_seed=0,
        config=config,
    )
    assert isinstance(runner, BeamSearchRunner)
    return runner


ACCELERATOR = Accelerator(cpu=True)

WRAPPED_MODELS = {
    "gpt2": WrappedGPT2Model(
        FakeModelForSequenceClassification(),  # type: ignore
        AutoTokenizer.from_pretrained("gpt2"),
        accelerator=ACCELERATOR,
    ),
    "bert": WrappedBERTModel(
        FakeModelForSequenceClassification(),  # type: ignore
        AutoTokenizer.from_pretrained("bert-base-uncased"),
        accelerator=ACCELERATOR,
    ),
    "pythia": WrappedGPTNeoXModel(
        FakeModelForSequenceClassification(),  # type: ignore
        AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped"),
        accelerator=ACCELERATOR,
    ),
}


def get_beam_search_runner(
    model_name: str,
    before_attack_text: str = "some before text",
    after_attack_text: str = "some after text",
    beam_search_width: int = 5,
) -> BeamSearchRunner:
    wrapped_model = WRAPPED_MODELS[model_name]
    return _get_runner(
        before_attack_text=before_attack_text,
        after_attack_text=after_attack_text,
        beam_search_width=beam_search_width,
        wrapped_model=wrapped_model,
    )


@pytest.fixture(params=WRAPPED_MODELS.keys())
def beam_search_runner(request) -> BeamSearchRunner:
    return get_beam_search_runner(request.param)


@pytest.mark.parametrize("model_name", WRAPPED_MODELS.keys())
@pytest.mark.parametrize("beam_search_width", [5, 13])
def test_n_best_candidates_to_keep(model_name: str, beam_search_width: int) -> None:
    runner = get_beam_search_runner(
        model_name=model_name, beam_search_width=beam_search_width
    )
    assert runner.n_best_candidates_to_keep == beam_search_width


# hypothesis was generating hex characters that messed up BERT tokenizer
text_no_specials = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126))


@given(initial_text=text_no_specials)
@pytest.mark.parametrize("num_initial_candidates", [1, 5])
@settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=500
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
