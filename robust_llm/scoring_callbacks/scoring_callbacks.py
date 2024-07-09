from typing import cast

import torch

from robust_llm.dist_utils import broadcast_list_of_bools, broadcast_tensor
from robust_llm.models.model_utils import (
    InferenceType,
    classification_losses_from_logits,
)
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.scoring_callbacks.scoring_callback_utils import (
    BinaryCallbackOutput,
    CallbackInput,
    CallbackRegistry,
    LabelData,
    TensorCallbackOutput,
    _classification_success_from_text,
    _classification_success_from_tokens,
    _generation_losses_from_embeds,
    _generation_losses_from_text,
    _generation_successes_from_text,
    _generation_successes_from_tokens,
    _output_generator_from_text,
    _validate_classification_labels,
    _validate_embeddings_input,
    _validate_text_input,
    _validate_tokens_input,
)
from robust_llm.scoring_callbacks.scoring_fn_utils import (
    BivariateScoringFn,
    ScoringFnReturnType,
    UnivariateScoringFn,
)


@CallbackRegistry.register_callback(name="successes_from_text", return_type="binary")
def successes_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
) -> BinaryCallbackOutput:
    """Compute the successes from the text input.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.

    Returns:
        list of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """

    input_data = _validate_text_input(callback_input.input_data)

    label_data: LabelData
    if victim.inference_type == InferenceType.CLASSIFICATION:
        if callback_input.clf_label_data is None:
            raise ValueError("Label data required for classification.")
        label_data = _validate_classification_labels(callback_input.clf_label_data)
        successes = _classification_success_from_text(victim, input_data, label_data)

    elif victim.inference_type == InferenceType.GENERATION:
        if callback_input.gen_target_data is None:
            raise ValueError("Target data required for generation.")
        label_data = _validate_text_input(callback_input.gen_target_data)
        successes = _generation_successes_from_text(victim, input_data, label_data)

    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")

    return BinaryCallbackOutput(successes=successes)


@CallbackRegistry.register_callback(name="successes_from_tokens", return_type="binary")
def successes_from_tokens_callback(
    victim: WrappedModel, callback_input: CallbackInput
) -> BinaryCallbackOutput:
    """Compute the successes from the token inputs.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.

    Returns:
        list of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """
    input_data = _validate_tokens_input(callback_input.input_data)
    label_data: LabelData
    if victim.inference_type == InferenceType.CLASSIFICATION:
        assert callback_input.clf_label_data is not None
        label_data = _validate_classification_labels(callback_input.clf_label_data)
        # We don't have attention masks for tokens, so we just use ones.
        # TODO(ian): Work out if we should pass an attention mask, or can assume
        # they're all unpadded.
        attention_mask = torch.ones_like(input_data)
        successes = _classification_success_from_tokens(
            victim, input_data, attention_mask, label_data
        )
    elif victim.inference_type == InferenceType.GENERATION:
        assert callback_input.gen_target_data is not None
        label_data = _validate_text_input(callback_input.gen_target_data)
        # Currently the _generation_successes_from_tokens function expects the
        # tokens to be a list of lists of ints, so we convert it here.
        # TODO(GH#441): Standardize this.
        list_input_data = input_data.tolist()
        successes = _generation_successes_from_tokens(
            victim=victim,
            prompt_input_ids=list_input_data,
            gen_target_data=label_data,
        )
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
    return BinaryCallbackOutput(successes=successes)


@CallbackRegistry.register_callback(name="losses_from_embeds", return_type="tensor")
def losses_from_embeds_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
    use_no_grad: bool = False,
) -> TensorCallbackOutput:
    """Compute the losses from the embeddings input.

    TODO(GH#440): Make this work for more batch sizes greater than 1.
    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
        use_no_grad: Whether to use torch.no_grad() when computing the losses.
    """
    input_data = _validate_embeddings_input(callback_input.input_data)
    label_data: LabelData
    if victim.inference_type == InferenceType.CLASSIFICATION:
        assert callback_input.clf_label_data is not None
        label_data = _validate_classification_labels(callback_input.clf_label_data)

        out = victim.classification_output_from_embeddings(
            input_ids=input_data["input_ids"],
            embeddings=input_data["embeddings"],
            use_no_grad=use_no_grad,
        )
        logits = out["logits"]
        assert victim.accelerator is not None
        assert isinstance(logits, torch.Tensor)
        losses = classification_losses_from_logits(logits, label_data)

    elif victim.inference_type == InferenceType.GENERATION:
        assert callback_input.gen_target_data is not None
        losses = _generation_losses_from_embeds(
            victim,
            input_data["input_ids"],
            input_data["embeddings"],
            callback_input.gen_target_data,
            use_no_grad=use_no_grad,
        )

    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
    return TensorCallbackOutput(losses=losses)


@CallbackRegistry.register_callback(name="losses_from_text", return_type="tensor")
def losses_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
) -> TensorCallbackOutput:
    """Compute losses from the text input.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
    """
    input_data = _validate_text_input(callback_input.input_data)
    label_data: LabelData
    if victim.inference_type == InferenceType.CLASSIFICATION:
        if callback_input.clf_label_data is None:
            raise ValueError("Label data required for classification.")
        label_data = _validate_classification_labels(callback_input.clf_label_data)

        # We use right-padding for non-autoregressive outputs.
        tokenized = victim.tokenize(
            input_data, return_tensors="pt", padding_side="right"
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        all_losses = []
        output_generator = victim.classification_output_from_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        batch_start = 0
        for out in output_generator:
            logits = out["logits"]
            assert victim.accelerator is not None
            assert isinstance(logits, torch.Tensor)
            batch_length = logits.shape[0]
            batch_label_data = label_data[batch_start : batch_start + batch_length]
            minibatch_losses = classification_losses_from_logits(
                logits, batch_label_data
            )
            all_losses.append(minibatch_losses)
            batch_start += batch_length
        losses = torch.cat(all_losses)

    elif victim.inference_type == InferenceType.GENERATION:
        if callback_input.gen_target_data is None:
            raise ValueError("Target data required for generation.")
        label_data = _validate_text_input(callback_input.gen_target_data)
        losses = _generation_losses_from_text(
            victim=victim,
            input_data=input_data,
            gen_target_data=label_data,
            use_no_grad=False,
        )
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")

    # Postconditions.
    assert losses.shape[0] == len(input_data)
    return TensorCallbackOutput(losses=losses)


@CallbackRegistry.register_callback(
    name="binary_univariate_fn_of_generation_from_text", return_type="binary"
)
def binary_univariate_fn_of_generation_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
    scoring_fn: UnivariateScoringFn,
) -> BinaryCallbackOutput:
    """Generates text from the victim and checks some binary property of that text.

    The property is defined by the scoring_fn function, which takes the
    generated text and returns a boolean. A simple example is checking whether
    the generated string contains an email address.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
        scoring_fn: A function that takes the generated text and returns a boolean.

    Returns:
        A BinaryCallbackOutput.
    """
    assert scoring_fn.return_type == ScoringFnReturnType.BOOL
    if victim.inference_type == InferenceType.CLASSIFICATION:
        raise ValueError(
            "fn_of_generation_from_text_callback is not supported for classification."
        )
    elif victim.inference_type == InferenceType.GENERATION:
        # TODO(GH#550): Clean up where we add the original input data
        if callback_input.original_input_data is not None:
            original_input_data = _validate_text_input(
                callback_input.original_input_data
            )
        else:
            original_input_data = None
        output_generator = _output_generator_from_text(victim, callback_input)
        all_successes = []
        all_generations = []
        batch_start = 0
        for outs in output_generator:
            assert victim.accelerator is not None
            # Only run scoring_fns on the main process.
            if victim.accelerator.is_main_process:
                batch_length = len(outs)
                batch_end = batch_start + batch_length

                if original_input_data is not None:
                    original_inputs = original_input_data[batch_start:batch_end]
                    outs = [
                        out.with_clean_input_text(orig)
                        for out, orig in zip(outs, original_inputs, strict=True)
                    ]

                successes = [scoring_fn(out) for out in outs]
                all_successes.extend(successes)
                # TODO(ian): Find a better way to record generations.
                gens = [out.get_full_text() for out in outs]
                all_generations.extend(gens)

                batch_start += batch_length

        # We have to type-check manually since scoring_fn could return floats or bools.
        assert all(isinstance(s, bool) for s in all_successes)
        successes_cast = cast(list[bool], all_successes)
        assert victim.accelerator is not None
        gathered_successes = broadcast_list_of_bools(successes_cast, victim.accelerator)
        # NOTE: It's fine not to gather the generations since we only log them
        # from the main process anyway.
        return BinaryCallbackOutput(
            successes=gathered_successes, info={"generations": all_generations}
        )
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")


@CallbackRegistry.register_callback(
    name="binary_bivariate_fn_of_generation_from_text", return_type="binary"
)
def binary_bivariate_fn_of_generation_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
    scoring_fn: BivariateScoringFn,
) -> BinaryCallbackOutput:
    """Generates text from victim and checks some property of that text and the target.

    The property is defined by the scoring_fn function, which takes the
    generated text as well as some target text and returns a boolean.
    A simple example is checking whether the generated string is exactly equal
    to the target.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
        scoring_fn: A function that takes the generated text *and* a target and
            returns a boolean.

    Returns:
        A BinaryCallbackOutput.
    """
    if victim.inference_type == InferenceType.CLASSIFICATION:
        raise ValueError(
            "fn_of_generation_from_text_callback is not supported for classification."
        )
    elif victim.inference_type == InferenceType.GENERATION:
        assert callback_input.gen_target_data is not None
        output_generator = _output_generator_from_text(victim, callback_input)

        if callback_input.original_input_data is not None:
            original_input_data = _validate_text_input(
                callback_input.original_input_data
            )
        else:
            original_input_data = None

        all_successes = []
        all_generations = []
        batch_start = 0
        gen_target_data = _validate_text_input(callback_input.gen_target_data)
        for outs in output_generator:
            assert victim.accelerator is not None
            # Only run scoring_fns on the main process.
            if victim.accelerator.is_main_process:
                batch_length = len(outs)
                batch_end = batch_start + batch_length

                # TODO(GH#550): Clean up where we add the original input data
                if original_input_data is not None:
                    original_inputs = original_input_data[batch_start:batch_end]
                    outs = [
                        out.with_clean_input_text(orig)
                        for out, orig in zip(outs, original_inputs, strict=True)
                    ]
                targets = gen_target_data[batch_start:batch_end]

                successes = [
                    scoring_fn(out, target)
                    for out, target in zip(outs, targets, strict=True)
                ]
                all_successes.extend(successes)
                # TODO(ian): Find a better way to record generations.
                gens = [out.get_full_text() for out in outs]
                all_generations.extend(gens)

                batch_start += batch_length

        # We have to type-check manually since scoring_fn could return floats or bools.
        assert all(isinstance(s, bool) for s in all_successes)
        successes_cast = cast(list[bool], all_successes)
        assert victim.accelerator is not None
        gathered_successes = broadcast_list_of_bools(successes_cast, victim.accelerator)
        # NOTE: It's fine not to gather the generations since we only log them
        # from the main process anyway.
        return BinaryCallbackOutput(
            successes=gathered_successes, info={"generations": all_generations}
        )
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")


@CallbackRegistry.register_callback(
    name="tensor_univariate_fn_of_generation_from_text", return_type="tensor"
)
def tensor_univariate_fn_of_generation_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
    scoring_fn: UnivariateScoringFn,
) -> TensorCallbackOutput:
    """Generates text from the victim and checks some binary property of that text.

    The property is defined by the scoring_fn function, which takes the
    generated text and returns a float. A simple example is checking how toxic
    the generated string is.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
        scoring_fn: A function that takes the generated text and returns a float.

    Returns:
        A TensorCallbackOutput.
    """
    assert scoring_fn.return_type == ScoringFnReturnType.FLOAT
    if victim.inference_type == InferenceType.CLASSIFICATION:
        raise ValueError(
            "fn_of_generation_from_text_callback is not supported for classification."
        )
    elif victim.inference_type == InferenceType.GENERATION:
        output_generator = _output_generator_from_text(victim, callback_input)

        if callback_input.original_input_data is not None:
            original_input_data = _validate_text_input(
                callback_input.original_input_data
            )
        else:
            original_input_data = None

        all_floats = []
        all_generations = []
        batch_start = 0
        for outs in output_generator:
            assert victim.accelerator is not None
            # Only run scoring_fns on the main process.
            if victim.accelerator.is_main_process:
                batch_length = len(outs)
                batch_end = batch_start + batch_length

                # TODO(GH#550): Clean up where we add the original input data
                if original_input_data is not None:
                    original_inputs = original_input_data[batch_start:batch_end]
                    outs = [
                        out.with_clean_input_text(orig)
                        for out, orig in zip(outs, original_inputs, strict=True)
                    ]

                floats = [scoring_fn(out) for out in outs]
                all_floats.extend(floats)
                # TODO(ian): Find a better way to record generations.
                gens = [out.get_full_text() for out in outs]
                all_generations.extend(gens)

                batch_start += batch_length

        assert victim.accelerator is not None
        gathered_floats = broadcast_tensor(torch.tensor(all_floats), victim.accelerator)
        # NOTE: It's fine not to gather the generations since we only log them
        # from the main process anyway.
        return TensorCallbackOutput(
            losses=gathered_floats, info={"generations": all_generations}
        )
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")


@CallbackRegistry.register_callback(
    name="tensor_bivariate_fn_of_generation_from_text", return_type="tensor"
)
def tensor_bivariate_fn_of_generation_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
    scoring_fn: BivariateScoringFn,
) -> TensorCallbackOutput:
    """Generates text from victim and checks some property of that text and the target.

    The property is defined by the scoring_fn function, which takes the
    generated text as well as optionally some target text and returns a float.
    A simple example is checking overlap between the generated string and the
    target (although we don't implement that here).

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
        scoring_fn: A function that takes the generated text *and* a target and
            returns a float.

    Returns:
        A TensorCallbackOutput.
    """
    if victim.inference_type == InferenceType.CLASSIFICATION:
        raise ValueError(
            "fn_of_generation_from_text_callback is not supported for classification."
        )
    elif victim.inference_type == InferenceType.GENERATION:
        assert callback_input.gen_target_data is not None
        output_generator = _output_generator_from_text(victim, callback_input)

        if callback_input.original_input_data is not None:
            original_input_data = _validate_text_input(
                callback_input.original_input_data
            )
        else:
            original_input_data = None

        all_floats = []
        all_generations = []
        batch_start = 0
        gen_target_data = _validate_text_input(callback_input.gen_target_data)
        for outs in output_generator:
            assert victim.accelerator is not None
            # Only run scoring_fns on the main process.
            if victim.accelerator.is_main_process:
                batch_length = len(outs)
                batch_end = batch_start + batch_length

                # TODO(GH#550): Clean up where we add the original input data
                if original_input_data is not None:
                    original_inputs = original_input_data[batch_start:batch_end]
                    outs = [
                        out.with_clean_input_text(orig)
                        for out, orig in zip(outs, original_inputs, strict=True)
                    ]
                targets = gen_target_data[batch_start:batch_end]

                floats = [
                    scoring_fn(out, target)
                    for out, target in zip(outs, targets, strict=True)
                ]
                all_floats.extend(floats)
                # TODO(ian): Find a better way to record generations.
                gens = [out.get_full_text() for out in outs]
                all_generations.extend(gens)

                batch_start += batch_length

        assert victim.accelerator is not None
        gathered_floats = broadcast_tensor(torch.tensor(all_floats), victim.accelerator)
        # NOTE: It's fine not to gather the generations since we only log them
        # from the main process anyway.
        return TensorCallbackOutput(
            losses=gathered_floats, info={"generations": all_generations}
        )
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
