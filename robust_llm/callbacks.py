from __future__ import annotations

from typing import TYPE_CHECKING

from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from typing_extensions import override

if TYPE_CHECKING:
    from robust_llm.training import Training


class GlobalTrainingStepRecordingWandbCallback(WandbCallback):
    """WandbCallback that sets up custom steps for wandb logging.

    Since it is hard to control the behavior of the default wandb's step,
    `wandb.run.step`, (e.g. logs coming from some libraries might increment it even when
    we don't want it), we have a convention in our codebase to use various "proxy step
    metrics", which are special wandb metrics that are used to define the x axis when
    plotting "regular" metrics.

    We define different steps for logs coming from different parts of the codebase, e.g.
    logs coming from the victim training, adversary training, etc. This way, when e.g.
    adversary is being trained and producing some logs, it doesn't affect the time as
    perceived by the victim training.

    We keep track of both number of updates and number of datapoints seen by the model,
    and we use the latter in defining steps.

    Additionally, the default HuggingFace Trainer's "global_step" is reset to 0 at the
    start of each `train` call. We want to record training progress over several rounds
    of iterative training, so we need a step/datapoint count that is not reset in this
    way. So we ignore HuggingFace's step and define our own steps.
    """

    def __init__(
        self,
        training: Training,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.training = training
        self.logging_counter = training.victim_training_logging_counter

    @override
    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)

        # Set the default horizontal axis on wandb
        # NOTE: Older versions of wandb don't have "define_metric",
        # so try updating wandb if you get an error here.
        assert hasattr(self._wandb, "define_metric")
        self._wandb.define_metric(
            "*", step_metric="victim_training_datapoint_count", step_sync=True
        )
        self._wandb.define_metric(
            "*",
            step_metric="victim_training_datapoint_count",
            step_sync=True,
        )

    @override
    def on_train_begin(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_begin(args, state, control, **kwargs)

        assert state.global_step is not None
        assert state.global_step == 0
        assert state.epoch is not None
        assert state.epoch == 0

        self.start_step = self.logging_counter.step_count
        self.start_datapoint = self.logging_counter.datapoint_count
        self.previous_global_step = 0

    @override
    def on_train_end(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_end(args, state, control, **kwargs)

        assert self.logging_counter.step_count == self.start_step + state.global_step

        # TODO: would be nice to have a way to check that we added the
        # correct number of datapoints too

    @override
    def on_step_end(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_end(args, state, control, **kwargs)

        assert state.global_step - self.previous_global_step == 1
        assert self.training.trainer is not None

        self.logging_counter.increment(
            step_count_to_add=1,
            datapoint_count_to_add=self.training.trainer.current_batch_size,
        )

        self.previous_global_step = state.global_step
