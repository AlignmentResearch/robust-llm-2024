import time
from dataclasses import dataclass

import wandb

from robust_llm import logger
from robust_llm.attacks.attack import AttackOutput
from robust_llm.logging_utils import wandb_log
from robust_llm.metrics.average_initial_breach import (
    AIBMetricResults,
    compute_aib_from_attack_output,
)
from robust_llm.metrics.iterations_for_success import (
    IFSMetricResults,
    compute_iterations_for_success,
)
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.scoring_callbacks.scoring_callback_utils import BinaryCallback


@dataclass
class RobustnessMetricResults:
    ifs_results: IFSMetricResults | None
    aib_results: AIBMetricResults | None

    @property
    def asr_per_iteration(self) -> list[float] | None:
        if self.ifs_results is None:
            return None
        return self.ifs_results.asr_per_iteration

    @property
    def ifs_per_decile(self) -> list[int | None] | None:
        if self.ifs_results is None:
            return None
        return self.ifs_results.ifs_per_decile

    @property
    def aib_per_decile(self) -> list[float] | None:
        if self.aib_results is None:
            return None
        return self.aib_results.aib_per_decile

    def unwrap_metrics(self, prefix: str = "metrics") -> dict[str, float | int | None]:
        metrics: dict[str, float | int | None] = {}
        if self.aib_per_decile is not None:
            for decile in range(11):
                metrics[f"{prefix}/aib@{decile/10}"] = self.aib_per_decile[decile]
        if self.ifs_per_decile is not None:
            for decile in range(11):
                metrics[f"{prefix}/ifs@{decile/10}"] = self.ifs_per_decile[decile]
        if self.asr_per_iteration is not None:
            total_iterations = len(self.asr_per_iteration)
            # Record ~10 iterations
            step = max(1, total_iterations // 10)
            for i in range(0, total_iterations, step):
                metrics[f"{prefix}/asr@{i}"] = self.asr_per_iteration[i]
            # Ensure the last iteration is always included
            if (total_iterations - 1) % step != 0:
                metrics[f"{prefix}/asr@{total_iterations-1}"] = self.asr_per_iteration[
                    -1
                ]
        return metrics

    def export_wandb_table(self):
        if self.asr_per_iteration is None:
            return
        table = wandb.Table(
            data=[self.asr_per_iteration],
            columns=[
                f"asr@{iteration}" for iteration in range(len(self.asr_per_iteration))
            ],
        )
        wandb_log({"asr_per_iteration": table}, commit=False)


def maybe_compute_robustness_metrics(
    compute_robustness_metric: bool,
    attack_out: AttackOutput,
    success_callback: BinaryCallback,
    model: WrappedModel,
) -> RobustnessMetricResults | None:
    if not compute_robustness_metric or attack_out.attack_data is None:
        return None
    # TODO(ian): Don't redundantly compute this and the ASR.
    # TODO(ian): Remove the try: except by making it work for all attacks.
    time_start = time.perf_counter()
    ifs_metric = None
    aib_metric = None
    try:
        ifs_metric = compute_iterations_for_success(
            attack_out=attack_out,
            success_callback=success_callback,
            model=model,
        )
    except Exception as e:
        logger.error(
            "Error computing robustness metric, might not be"
            f" implemented for this attack yet: {e}"
        )
    try:
        aib_metric = compute_aib_from_attack_output(attack_out)
    except Exception as e:
        logger.error(
            "Error computing average initial breach, might not be"
            f" implemented for this attack yet: {e}"
        )
    results = RobustnessMetricResults(ifs_results=ifs_metric, aib_results=aib_metric)
    time_end = time.perf_counter()
    logger.info(
        f"Time taken for robustness metrics computation: {time_end - time_start:.2f}s"
    )
    return results
