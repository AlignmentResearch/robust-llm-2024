from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class CallbackConfig:
    """Configs used by ScoringCallbacks.

    Args:
        callback_name: str
            The name of the callback function in the CallbackRegistry.
        callback_return_type: str
            The return type of the callback function, telling us which sub-registry
            to look in. Should match an item in the CallbackReturnType enum.
    """

    callback_name: str = MISSING
    callback_return_type: str = MISSING


@dataclass
class AutoregressiveCallbackConfig(CallbackConfig):
    """Configs used by ScoringCallbacks with text generation.

    Args:
        scoring_fn: str
            The name of the scoring function in the ScoringFnRegistry.
        scoring_fn_arity: int
            The number of arguments the scoring function takes, telling us which
            sub-registry to look in. Should be 1 or 2.
    """

    scoring_fn: str = MISSING
    scoring_fn_arity: int = MISSING


cs = ConfigStore.instance()
cs.store(name="DEFAULT", group="callback", node=CallbackConfig)
cs.store(name="AUTOREGRESSIVE", group="callback", node=AutoregressiveCallbackConfig)
