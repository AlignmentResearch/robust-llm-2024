import wandb
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from typing_extensions import override


class CrossTrainRunStepRecordingWandbCallback(WandbCallback):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_past_training_steps_completed: int = 0

    @override
    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)

        # Undo setting "global_step" (which resets each `train`) as the default
        # step metric
        # NOTE: apparently older versions of wandb don't have "define_metric"
        assert hasattr(self._wandb, "define_metric")
        self._wandb.define_metric(
            "*", step_metric="overall_global_step", step_sync=True
        )

    @override
    def on_train_end(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_end(args, state, control, **kwargs)

        self.num_past_training_steps_completed += state.global_step

    @override
    def on_step_begin(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_begin(args, state, control, **kwargs)

        wandb.log(
            {
                "overall_global_step": self.num_past_training_steps_completed
                + state.global_step
            },
            commit=False,
        )
