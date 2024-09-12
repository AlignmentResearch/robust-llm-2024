from unittest.mock import patch

import pytest
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.attacks.search_based.runners import GCGRunner, make_runner
from robust_llm.attacks.search_based.utils import (
    PreppedExample,
    PromptTemplate,
    ReplacementCandidate,
)
from robust_llm.config import GCGAttackConfig
from robust_llm.models import GPTNeoXModel
from robust_llm.models.model_utils import InferenceType

# We use a real Accelerator for this test despite the fact that it
# might lead to flaky tests on CircleCI because it's really hard to
# mock the Accelerator.backward method that we need.
ACCELERATOR = Accelerator()


# TODO(GH#862): Add similar tests for other non-pythia tokenizers.
@pytest.mark.xfail(
    strict=False,  # strict=False is necessary because it doesn't fail consistently.
    reason="This test is flaky on CI because of the Accelerator.",
)
def test_shape_mismatch():
    # we need a model for pythia because we access the config
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token

    wrapped_model = GPTNeoXModel(
        model,
        tokenizer,
        accelerator=ACCELERATOR,
        inference_type=InferenceType("classification"),
        train_minibatch_size=2,
        eval_minibatch_size=2,
        generation_config=None,
        family="pythia",
    )
    config = GCGAttackConfig(
        n_candidates_per_it=1,
        n_attack_tokens=10,
        top_k=1,
    )

    prompt_template = PromptTemplate(
        before_attack="You will be given a review below to classify based on its sentiment. The review will be either positive or negative. If the review is positive, return POSITIVE. If the review is negative, return NEGATIVE.\n\nAs far as I know this was my first experience with Icelandic movies. It's such a relief to see something else than your regular Hollywood motion picture. Too bad that movies like this one have a small chance of succeeding in the big world. I can only hope that people watch this by accident, by recommendation or other...<br /><br />Because it's really worth while. I left the cinema feeling really sad. I couldn't get the tragic destiny's of the characters out of my head. And it impressed me even more when I thought of the complexity of the film. Not only was it a tragic story, it had excellent comic reliefs and a very good soundtrack.<br /><br />If you have the opportunity, watch it! It's really thought provoking and made me ponder a lot.<br /><br />",  # noqa: E501
        after_attack="\n\nAnswer:",
    )

    prepped_examples = [
        PreppedExample(
            prompt_template=prompt_template,
            clf_label=1,
            gen_target=" POSITIVE",
        )
    ]
    runner = make_runner(
        victim=wrapped_model,
        prepped_examples=prepped_examples,
        random_seed=0,
        n_its=2,
        config=config,
    )
    assert isinstance(runner, GCGRunner)

    attack_text = "']{}(ifdef Filed HOLDERSGradeSyscallpgen }^{ FAQ"

    def mock_get_candidate_texts_and_replacements(
        self,
        candidate_texts,
        original_fn=GCGRunner._get_candidate_texts_and_replacements,
    ):
        # We call the original because we need it to potentially raise an error,
        # but we don't care about the output.
        original_fn(self, candidate_texts)
        return [
            (
                attack_text,
                # This replaces `]{}(` with `textbf`
                ReplacementCandidate(1, 11765),
            )
        ]

    with patch.object(
        GCGRunner,
        "_get_candidate_texts_and_replacements",
        mock_get_candidate_texts_and_replacements,
    ):
        runner.run()
