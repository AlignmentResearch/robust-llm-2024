from transformers.integrations import WandbCallback
from typing_extensions import override


class CustomWandbCallback(WandbCallback):
    @override
    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        
        # Try to undo the setting "global_step" as the default step metric
        if getattr(self._wandb, "define_metric", None):
            self._wandb.define_metric("*", step_metric="Step", step_sync=True)
