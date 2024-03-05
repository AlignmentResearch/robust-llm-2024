from __future__ import annotations

from typing import TYPE_CHECKING

from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from typing_extensions import override

if TYPE_CHECKING:
    from robust_llm.training import Training

from robust_llm.logging_utils import setup_wandb_metrics


class CustomLoggingWandbCallback(WandbCallback):
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

        # We wait until now to set up the wandb step metrics
        # (which logged values can be used as x-axes,
        # and what the default x-axes are set to be)
        # because when we initialize the HuggingFace Trainer,
        # it sets up its own `global_step` default metric, which would
        # overwrite anything we had set earlier. Thus, we wait until
        # that happens and then set up our own metrics.
        setup_wandb_metrics()

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
    ) -> None:
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

        # We do not commit here, since the train loss and other stats are
        # logged by the HuggingFace Trainer _after_ `on_step_end` is called,
        # so we would be logging with an off-by-one error if we committed here.
        # Instead, we rely on the HuggingFace Trainer to commit the logs at the
        # same time that it commits the train loss and other stats.
        # In the case where we `logging_steps` > 1, this means that we
        # will call `wandb.log` with `commit=False` multiple times before
        # committing. This is okay because while wandb will overwrite
        # the old counts with the new ones, when it is time to commit,
        # the currently stored counts will be the ones that correctly
        # correspond to the current train loss and other stats.
        self.logging_counter.increment(
            step_count_to_add=1,
            datapoint_count_to_add=self.training.trainer.current_batch_size,
            commit=False,
        )

        self.previous_global_step = state.global_step
