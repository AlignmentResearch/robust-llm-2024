"""
ASR per iteration metric:

Simply the attack success rate at each iteration of the attack.
"""

import time
from dataclasses import dataclass

from robust_llm import logger
from robust_llm.attacks.attack import AttackOutput
from robust_llm.metrics.metric_utils import (
    _compute_clf_asr_from_logits,
    _dataset_for_iteration,
    get_attack_output_from_wandb,
)
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.scoring_callbacks.scoring_callback_utils import (
    BinaryCallback,
    CallbackInput,
)


@dataclass(frozen=True)
class ASRMetricResults:
    """Tracks the ASR at each iteration.

    NOTE: iterations are 1-indexed. This is because the first iteration is the
    original un-attacked dataset.
    """

    asr_per_iteration: list[float]

    def interpolated_iteration_for_asr(self, asr_threshold: float) -> float | None:
        """Returns the interpolated iteration at which the ASR would cross a threshold.

        See docstring for `interpolated_iteration_for_asr` for more details.
        """
        return interpolated_iteration_for_asr(self.asr_per_iteration, asr_threshold)


def interpolated_iteration_for_asr(
    asr_per_iteration: list[float], asr_threshold: float
) -> float | None:
    """Returns the interpolated iteration at which the ASR would cross a threshold.

    We linearly interpolate between the first two iterations on either side
    of the desired threshold.

    TODO(ian): Maybe do more sophisticated interpolation.

    Args:
        asr_per_iteration: The ASR at each iteration.
        asr_threshold: The threshold to cross.

    Returns:
        The interpolated iteration at which the ASR would cross the threshold
        If the ASR never crosses the threshold, returns None.
    """
    if asr_threshold == 0:
        return 0.0

    prev_asr = 0.0
    for i, asr in enumerate(asr_per_iteration):
        if asr == asr_threshold:
            return i
        if asr > asr_threshold:
            fraction_between_asrs = (asr_threshold - prev_asr) / (asr - prev_asr)
            assert fraction_between_asrs > 0, "does ASR start at 0?"
            interpolated_iteration = (i - 1) + fraction_between_asrs
            return interpolated_iteration
        prev_asr = asr

    return None


def compute_asr_per_iteration(
    attack_out: AttackOutput,
    success_callback: BinaryCallback,
    model: WrappedModel,
) -> ASRMetricResults:
    """Computes the ASR per iteration of the attack.

    Args:
        attack_out: The AttackOutput object from the attack.
        success_callback: The callback to use to evaluate the attack.
        model: The model to evaluate the attack on.

    Returns:
        An object containing the ASRs for each iteration of the attack, and the
        iteration number at which the ASR crosses each decile. The decile is None
        if the ASR never crosses that threshold.
    """
    dataset = attack_out.dataset
    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None
    logits = attack_out.attack_data.logits  # [n_val, n_its, n_labels]

    if dataset.inference_type == InferenceType.CLASSIFICATION and logits is not None:
        results = compute_asr_per_iteration_from_logits(
            attack_out=attack_out,
        )
    else:
        results = compute_asr_per_iteration_from_text(
            attack_out=attack_out,
            success_callback=success_callback,
            model=model,
        )
    return results


def compute_asr_per_iteration_from_text(
    attack_out: AttackOutput,
    success_callback: BinaryCallback,
    model: WrappedModel,
) -> ASRMetricResults:
    dataset = attack_out.dataset
    # We use dataset_to_attack so that we use the same examples as in attacked_dataset
    original_input_data = model.maybe_apply_user_template(dataset.ds["text"])

    # Somewhat hacky way to get the number of iterations
    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None
    n_its = len(attack_out.attack_data.iteration_texts[0])
    asrs = []

    # Special handling for 'iteration' 0, which is the un-attacked inputs.
    asrs.append(0.0)

    for iteration in range(n_its):
        ds = _dataset_for_iteration(attack_out, iteration)

        iteration_in = CallbackInput(
            input_data=ds["text"],
            original_input_data=original_input_data,
            clf_label_data=ds["clf_label"],
            gen_target_data=ds["gen_target"],
        )
        iteration_out = success_callback(model, iteration_in)
        iteration_n_examples = len(iteration_out.successes)
        # ASR is 1 - accuracy, i.e. the fraction of examples where the model is
        # not successful.
        iteration_asr = iteration_out.successes.count(False) / iteration_n_examples

        asrs.append(iteration_asr)
        # Print the ASR for this iteration every roughly 10% of the way
        logging_step = max(1, n_its // 10)
        if iteration % logging_step == 0:
            logger.info(f"ASR for iteration {iteration+1}: {iteration_asr}")

    assert len(asrs) == n_its + 1
    return ASRMetricResults(asr_per_iteration=asrs)


def compute_asr_per_iteration_from_logits(
    attack_out: AttackOutput,
) -> ASRMetricResults:

    # TODO(ian): Remove these asserts
    assert attack_out.attack_data is not None
    logits = attack_out.attack_data.logits  # [n_val, n_its, n_labels]
    assert logits is not None

    # Somewhat hacky way to get the number of iterations
    n_its = len(attack_out.attack_data.iteration_texts[0])
    asrs = []

    # Special handling for 'iteration' 0, which is the un-attacked inputs.
    asrs.append(0.0)

    for iteration in range(n_its):
        ds = _dataset_for_iteration(attack_out, iteration)
        clf_labels = ds["clf_label"]
        iteration_logits = [logits[i][iteration] for i in range(len(logits))]
        # TODO: Fix type hinting on logits
        iteration_asr = _compute_clf_asr_from_logits(
            iteration_logits, clf_labels  # type: ignore
        )

        asrs.append(iteration_asr)
        # Print the ASR for this iteration every roughly 10% of the way
        logging_step = max(1, n_its // 10)
        if iteration % logging_step == 0:
            logger.info(f"ASR for iteration {iteration+1}: {iteration_asr}")

    assert len(asrs) == n_its + 1
    results = ASRMetricResults(asr_per_iteration=asrs)
    return results


def compute_asr_per_iteration_from_wandb(
    group_name: str, run_index: str
) -> ASRMetricResults:
    attack_output = get_attack_output_from_wandb(group_name, run_index)

    tic = time.perf_counter()
    results = compute_asr_per_iteration_from_logits(attack_output)
    toc = time.perf_counter()
    print(results)
    print(f"Computed ASR per iteration metric in {toc - tic:.2f} seconds")
    return results
