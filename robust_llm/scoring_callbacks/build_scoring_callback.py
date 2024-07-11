from functools import partial

from robust_llm.config.callback_configs import (
    AutoregressiveCallbackConfig,
    CallbackConfig,
)
from robust_llm.scoring_callbacks.scoring_callback_utils import (
    BinaryCallback,
    CallbackRegistry,
    CallbackReturnType,
    TensorCallback,
)
from robust_llm.scoring_callbacks.scoring_fn_utils import ScoringFn, ScoringFnReturnType
from robust_llm.scoring_callbacks.scoring_fns import ScoringFnRegistry


def build_binary_scoring_callback(config: CallbackConfig) -> BinaryCallback:
    """Builds a binary scoring callback from a CallbackConfig."""
    callback_return_type = CallbackReturnType(config.callback_return_type)
    assert callback_return_type == CallbackReturnType.BINARY
    callback_fn = CallbackRegistry.get_binary_callback(config.callback_name)

    # Currently only Autoregressive callback fns have scoring_fns
    if isinstance(config, AutoregressiveCallbackConfig):
        scoring_fn: ScoringFn
        match config.scoring_fn_arity:
            case 1:
                scoring_fn = ScoringFnRegistry.get_univariate_scoring_fn(
                    config.scoring_fn
                )
            case 2:
                scoring_fn = ScoringFnRegistry.get_bivariate_scoring_fn(
                    config.scoring_fn
                )
            case _:
                raise ValueError(f"Unknown scoring_fn_arity: {config.scoring_fn_arity}")
        assert scoring_fn.return_type == ScoringFnReturnType.BOOL
        # TODO(ian): Work out type hints for partial here
        return partial(
            callback_fn, scoring_fn=scoring_fn  # pyright: ignore[reportCallIssue]
        )
    else:
        return callback_fn


def build_tensor_scoring_callback(config: CallbackConfig) -> TensorCallback:
    """Builds a tensor scoring callback from a CallbackConfig."""
    callback_return_type = CallbackReturnType(config.callback_return_type)
    assert callback_return_type == CallbackReturnType.TENSOR
    callback_fn = CallbackRegistry.get_tensor_callback(config.callback_name)

    # Currently only Autoregressive callback fns have scoring_fns
    if isinstance(config, AutoregressiveCallbackConfig):
        assert isinstance(config, AutoregressiveCallbackConfig)
        scoring_fn: ScoringFn
        match config.scoring_fn_arity:
            case 1:
                scoring_fn = ScoringFnRegistry.get_univariate_scoring_fn(
                    config.scoring_fn
                )
            case 2:
                scoring_fn = ScoringFnRegistry.get_bivariate_scoring_fn(
                    config.scoring_fn
                )
            case _:
                raise ValueError(f"Unknown scoring_fn_arity: {config.scoring_fn_arity}")
        assert scoring_fn.return_type == ScoringFnReturnType.FLOAT
        # TODO(ian): Work out type hints for partial here
        return partial(
            callback_fn, scoring_fn=scoring_fn  # pyright: ignore[reportCallIssue]
        )
    else:
        return callback_fn


def build_scoring_callback(config: CallbackConfig) -> BinaryCallback | TensorCallback:
    """Builds a scoring callback from a CallbackConfig."""
    callback_return_type = CallbackReturnType(config.callback_return_type)
    if callback_return_type == CallbackReturnType.BINARY:
        return build_binary_scoring_callback(config)
    elif callback_return_type == CallbackReturnType.TENSOR:
        return build_tensor_scoring_callback(config)
    else:
        raise ValueError(f"Unknown callback_return_type: {config.callback_return_type}")
