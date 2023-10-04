from transformers.integrations import WandbCallback
from typing_extensions import override

import wandb

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class CrossTrainRunStepRecordingWandbCallback(WandbCallback):
    
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_past_training_steps_completed: int = 0
    
    @override
    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        
        # Try to undo the setting "global_step" as the default step metric
        if getattr(self._wandb, "define_metric", None):
            self._wandb.define_metric("*", step_metric="manual_global_step", step_sync=True)
    
    @override
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        
        self.num_past_training_steps_completed += state.global_step

    @override
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_begin(args, state, control, **kwargs)
        
        wandb.log({"overall_cross_round_global_step": self.num_past_training_steps_completed + state.global_step})
