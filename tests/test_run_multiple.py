import pytest

from robust_llm.batch_job_utils import run_multiple


def test_clean_example():
    EXPERIMENT_NAME = "typo_example"
    HYDRA_CONFIG = "simple_adv_eval_tensor_trust"

    MODELS = [
        "AlignmentResearch/robust_llm_pythia-tt-14m-mz-v0",
    ]
    OVERRIDE_ARGS_LIST = [
        {
            "experiment.environment.model_name_or_path": model,
        }
        for model in MODELS
    ]
    # check it runs without errors
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
        dry_run=True,
        skip_git_checks=True,
    )


def test_typo_example():
    EXPERIMENT_NAME = "typo_example"
    HYDRA_CONFIG = "simple_adv_eval_tensor_trust"

    MODELS = [
        "AlignmentResearch/robust_llm_pythia-tt-14m-mz-v0",
    ]
    OVERRIDE_ARGS_LIST = [
        {
            # very obvious mistake in the overrides
            "experiment.environment.model_name_or_path_OOPS_typo": model,
        }
        for model in MODELS
    ]
    with pytest.raises(ValueError) as e_info:
        run_multiple(
            EXPERIMENT_NAME,
            HYDRA_CONFIG,
            OVERRIDE_ARGS_LIST,
            memory="50G",
            dry_run=True,
            skip_git_checks=True,
        )
    assert str(e_info.value) == "override_args are invalid, aborting."
