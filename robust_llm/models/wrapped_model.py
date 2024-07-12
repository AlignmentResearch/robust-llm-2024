from __future__ import annotations

import copy
import dataclasses
import shutil
import tempfile
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Optional, overload

import torch
import torch.distributed
import transformers
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.modeling_outputs import ModelOutput

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import (
    AutoregressiveOutput,
    InferenceType,
    SuppressPadTokenWarning,
    build_dataloader,
    dict_to_device,
    get_num_parameters,
    load_hf_model,
    maybe_no_grad,
    prepare_model_with_accelerate,
    remove_padding_tokens,
)
from robust_llm.models.prompt_templates import PromptTemplate, PromptTemplateBuilder
from robust_llm.utils import is_correctly_padded


class WrappedModel(ABC):
    """Combines a model and a tokenizer."""

    _registry: dict[str, type[WrappedModel]] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        family: str,
        generation_config: GenerationConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize a WrappedModel.

        Args:
            model: The model to wrap.
            right_tokenizer: The tokenizer to use. Should be right-padded (the
                left-padded version will be loaded if needed.)
            accelerator: The accelerator to use.
            inference_type: The type of inference this model is for ('generation'
                or 'classification' or 'trl')
            train_minibatch_size: The minibatch size to use for training.
            eval_minibatch_size: The minibatch size to use for evaluation.
            family: The family of the model, useful for logging model details
                to wandb alongside experiment results.
            generation_config: The generation config to use for generation.
            system_prompt: The system prompt to use for chat models.
                If None, the default system prompt will be used.
        """
        # We need to compute the number of parameters before any accelerate preparation
        # because the model will be sharded across devices.
        # NOTE: If we switch to loading directly to devices, we'll need to change how we
        # compute the number of parameters.
        self._n_params = get_num_parameters(model)
        self.family = family
        self.accelerator = accelerator
        if self.accelerator is not None:
            self.model = prepare_model_with_accelerate(self.accelerator, model)
        else:
            self.model = model

        self.right_tokenizer = right_tokenizer
        self.inference_type = inference_type
        self.train_minibatch_size = train_minibatch_size
        self.eval_minibatch_size = eval_minibatch_size
        self.generation_config = generation_config
        self.system_prompt = system_prompt

    @property
    def n_params(self) -> int:
        return self._n_params

    def push_to_hub(
        self,
        repo_id: str,
        revision: Optional[str],
        retries: int,
        cooldown_seconds: float,
        local_dir: Path | None = None,
    ):
        """Pushes the model and tokenizer to the hub with retries."""
        assert self.accelerator is not None
        for attempt in range(retries):
            try:
                self._push_to_hub_once(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=local_dir,
                )
                return
            except Exception as e:
                warnings.warn(
                    f"Failed to push to hub on attempt {attempt + 1} of {retries}: "
                    f"{e}\n{traceback.format_exc()}",
                    stacklevel=2,
                )
            if attempt + 1 < retries:
                time.sleep(cooldown_seconds)
        raise RuntimeError(f"Failed to push to hub after {retries} attempts.")

    def save_local(
        self,
        output_dir: Path,
    ):
        """Save the model and tokenizer to a local directory"""
        assert self.accelerator is not None
        state_dict = self.accelerator.get_state_dict(self.model)
        self.model.save_pretrained(
            save_directory=output_dir,
            save_function=self.accelerator.save,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
            safe_serialization=False,
        )
        self.right_tokenizer.save_pretrained(
            save_directory=output_dir,
        )

    def _push_to_hub_once(
        self,
        repo_id: str,
        revision: Optional[str],
        local_dir: Path | None = None,
    ):
        """Makes one attempt to push the model and tokenizer to the hub.

        Args:
            repo_id: The ID of the huggingface hub repo to push to.
            revision: The revision to push to.
            local_dir: The local directory to save the model and tokenizer to.
                If None, a temporary directory will be created.
        """
        save_dir = local_dir or Path(tempfile.mkdtemp())
        self.save_local(save_dir)

        # Now separately push to hub, using an internal transformers
        # function. We have to use an internal function because both
        # `push_to_hub` and `save_pretrained` do not give enough flexibility
        # when using FSDP: `push_to_hub` doesn't let you pass in a
        # state_dict and `save_pretrained` doesn't let you pass in a
        # tag/revision for the repo.
        assert self.accelerator is not None
        if self.accelerator.is_main_process:
            repo_id = self.model._create_repo(repo_id)
            revision = revision or "main"
            self.model._upload_modified_files(
                working_dir=save_dir,
                repo_id=repo_id,
                files_timestamps=dict(),
                commit_message="Pushing model and tokenizer to hub",
                revision=revision,  # type: ignore  # bad hinting in transformers
            )

            # Only clean up if we didn't want to keep the local directory.
            if local_dir is None:
                shutil.rmtree(save_dir)

    @classmethod
    def register_subclass(cls, name):
        """Registers a subclass of WrappedModel.

        We use this so we can create subclasses of WrappedModel without circular
        imports.

        (From https://chat.openai.com/share/e/162dd905-0ce9-4981-b1a7-b0d0306ea99b)
        """

        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        accelerator: Accelerator | None,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> WrappedModel:
        """Creates a WrappedModel from a ModelConfig."""
        inference_type = InferenceType(config.inference_type)
        model = load_hf_model(
            name_or_path=config.name_or_path,
            revision=config.revision,
            inference_type=inference_type,
            strict_load=config.strict_load,
            torch_dtype=getattr(torch, config.dtype),
            num_classes=num_classes,
        )

        try:
            subcls = cls._registry[config.family]
        except KeyError:
            raise ValueError(f"Unsupported model family: {config.family}")

        train_mb_size = int(config.train_minibatch_size * config.minibatch_multiplier)
        eval_mb_size = int(config.eval_minibatch_size * config.minibatch_multiplier)
        # Loads the tokenizer with right padding. We'll load the tokenizer
        # with left padding lazily if we need it.
        right_tokenizer = subcls.load_tokenizer(config)
        return subcls(
            model=model,
            right_tokenizer=right_tokenizer,
            accelerator=accelerator,
            inference_type=inference_type,
            train_minibatch_size=train_mb_size,
            eval_minibatch_size=eval_mb_size,
            generation_config=config.generation_config,
            family=config.family,
            system_prompt=config.system_prompt,
        )

    def classification_output_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_no_grad: bool = True,
        minibatch_size: int | None = None,
    ) -> Iterator[ModelOutput]:
        """Returns the classification logits from the token ids.

        Args:
            input_ids: The token ids to calculate the logits on.
            attention_mask: The attention mask for the input_ids. Only needed
                if the input_ids are actually padded.
            use_no_grad: Whether to use torch.no_grad(). Defaults to True
                because usually we do not need gradient information, but it is
                needed for gradient-based attacks like GCG. Additionally, this will
                fail loudly if we try to backpropagate through it, whereas False
                will just be silently inefficient.
            minibatch_size: The minibatch size to use. If None, we use
                self.eval_minibatch_size.

        Yields:
            A SequenceClassifierOutput object, which has a 'logits' attribute.
        """

        minibatch_size = minibatch_size or self.eval_minibatch_size

        dataloader = build_dataloader(
            input_ids=input_ids,
            attention_mask=attention_mask,
            minibatch_size=minibatch_size,
        )
        assert self.accelerator is not None
        dataloader = self.accelerator.prepare(dataloader)

        # TODO (ian): Maybe I need an accelerator.gather_for_metrics somewhere here?
        with maybe_no_grad(use_no_grad):
            for minibatch in dataloader:
                minibatch_out = self(
                    input_ids=minibatch["input_ids"],
                    attention_mask=minibatch["attention_mask"],
                )
                minibatch_out = self.accelerator.gather_for_metrics(minibatch_out)
                assert isinstance(minibatch_out, ModelOutput)
                yield minibatch_out

    def classification_output_from_embeddings(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        use_no_grad: bool = True,
    ) -> ModelOutput:
        """Returns the classification logits from the embeddings.

        TODO (ian): Make this run on batch size >1.
        Args:
            input_ids: The token ids to calculate the logits on.
                These are needed to check for cache hits.
            embeddings: The embeddings to calculate the logits on.
            use_no_grad: Whether to use torch.no_grad(). Defaults to True
                because usually we do not need gradient information, but it is
                needed for gradient-based attacks like GCG. Additionally, this will
                fail loudly if we try to backpropagate through it, whereas False
                will just be silently inefficient.

        Returns:
            A SequenceClassifierOutput object, which has a 'logits' attribute.
        """

        if embeddings.shape[0] != 1:
            raise ValueError("This method currently only works for batch size 1.")
        assert embeddings.shape[0] == input_ids.shape[0]
        assert embeddings.shape[1] == input_ids.shape[1]

        with maybe_no_grad(use_no_grad):
            with SuppressPadTokenWarning(self.model):
                out = self(
                    input_ids=input_ids,
                    inputs_embeds=embeddings,
                )
        return out

    def generation_output_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_no_grad: bool = True,
        minibatch_size: int | None = None,
    ) -> Iterator[ModelOutput]:
        """Returns the generation logits from the token ids.

        Args:
            input_ids: The token ids to calculate the logits on.
            attention_mask: The attention mask for the input_ids. Only needed
                if the input_ids are actually padded.
            use_no_grad: Whether to use torch.no_grad(). Defaults to True
                because usually we do not need gradient information, but it is
                needed for gradient-based attacks like GCG. Additionally, this will
                fail loudly if we try to backpropagate through it, whereas False
                will just be silently inefficient.
            minibatch_size: The minibatch size to use. If None, we use
                self.eval_minibatch_size.

        Yields:
            A SequenceClassifierOutput object, which has a 'logits' attribute.
        """

        minibatch_size = minibatch_size or self.eval_minibatch_size

        dataloader = build_dataloader(
            input_ids=input_ids,
            attention_mask=attention_mask,
            minibatch_size=minibatch_size,
        )
        assert self.accelerator is not None
        dataloader = self.accelerator.prepare(dataloader)

        # TODO (ian): Maybe I need an accelerator.gather_for_metrics somewhere here?
        with maybe_no_grad(use_no_grad):
            for minibatch in dataloader:
                minibatch_out = self(
                    input_ids=minibatch["input_ids"],
                    attention_mask=minibatch["attention_mask"],
                )
                minibatch_out = self.accelerator.gather_for_metrics(minibatch_out)
                assert isinstance(minibatch_out, ModelOutput)
                yield minibatch_out

    def generation_output_from_embeddings(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        use_no_grad: bool = True,
    ) -> Iterator[ModelOutput]:
        """Returns the classification logits from the embeddings.

        TODO (ian): Make this run on batch size >1.
        Args:
            input_ids: The token ids to calculate the logits on.
                These are needed to check for cache hits.
            embeddings: The embeddings to calculate the logits on.
            use_no_grad: Whether to use torch.no_grad(). Defaults to True
                because usually we do not need gradient information, but it is
                needed for gradient-based attacks like GCG. Additionally, this will
                fail loudly if we try to backpropagate through it, whereas False
                will just be silently inefficient.

        Yields:
            A SequenceClassifierOutput object, which has a 'logits' attribute.
        """

        if embeddings.shape[0] != 1:
            raise ValueError("This method currently only works for batch size 1.")
        assert embeddings.shape[0] == input_ids.shape[0]
        # We don't need these to be equal because the input_ids are just for cache.
        assert embeddings.shape[1] >= input_ids.shape[1]

        with maybe_no_grad(use_no_grad):
            with SuppressPadTokenWarning(self.model):
                out = self(
                    input_ids=input_ids,
                    inputs_embeds=embeddings,
                )
                yield out

    def autoregressive_generation_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        minibatch_size: int | None = None,
    ) -> Iterator[list[AutoregressiveOutput]]:
        """Returns the autoregressive generation from the token ids.

        Args:
            input_ids: The token ids to start the generation from.
            attention_mask: The attention mask for the input_ids. Only needed
                if the input_ids are actually padded.
            minibatch_size: The minibatch size to use. If None, we use
                self.eval_minibatch_size.

        Yields:
            A list of strings, which are the generated sequences.
        """
        assert self.inference_type == InferenceType.GENERATION
        if attention_mask is not None:
            if not is_correctly_padded(attention_mask, "left"):
                raise ValueError(
                    "It seems like your inputs are not correctly left-padded."
                )

        minibatch_size = minibatch_size or self.eval_minibatch_size

        dataloader = build_dataloader(
            input_ids=input_ids,
            attention_mask=attention_mask,
            minibatch_size=minibatch_size,
        )
        assert self.accelerator is not None
        dataloader = self.accelerator.prepare(dataloader)

        with maybe_no_grad(use_no_grad=True):
            for minibatch in dataloader:
                minibatch_tokens = self.generate(
                    input_ids=minibatch["input_ids"],
                    attention_mask=minibatch["attention_mask"],
                    generation_config=self.generation_config,
                )
                # Very important to pad_across_processes before gathering;
                # otherwise it'll silently hang on one process if ever the
                # returned tensors have a size mismatch.
                minibatch_tokens = self.accelerator.pad_across_processes(
                    minibatch_tokens, dim=1
                )
                minibatch_tokens = self.accelerator.gather_for_metrics(minibatch_tokens)
                assert isinstance(minibatch_tokens, torch.Tensor)
                # Separate out the input and output tokens.
                input_tokens = minibatch_tokens[:, : input_ids.shape[1]]
                output_tokens = minibatch_tokens[:, input_ids.shape[1] :]

                input_texts = self.decode_and_unpad(
                    input_tokens, skip_special_tokens=False
                )

                # For now, we don't skip special tokens, because that removes the
                # chat formatting tokens.
                # TODO(ian): Work out how to handle special tokens in the output.
                output_texts = self.decode_and_unpad(
                    output_tokens, skip_special_tokens=False
                )
                yield [
                    AutoregressiveOutput(in_text, out_text)
                    for in_text, out_text in zip(input_texts, output_texts, strict=True)
                ]

    def generate_from_text(self, text: str) -> list[int]:
        """Returns the autoregressive generation from the text."""
        inputs = self.tokenize(
            text,
            return_tensors="pt",
            # We use left-padding for autoregressive outputs.
            padding_side="left",
        )
        inputs = inputs.to(device=self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        assert input_ids.shape[0] == 1
        assert attention_mask.shape[0] == 1
        all_tokens = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.right_tokenizer.pad_token_id,
        )
        assert isinstance(all_tokens, torch.Tensor)
        assert all_tokens.shape[0] == 1
        output_tokens = all_tokens[0, input_ids.shape[1] :]

        return output_tokens.cpu().tolist()

    def __call__(self, **inputs):
        return self.forward(**inputs)

    def forward(self, **inputs):
        # If we have both inputs_embeds and input_ids, we drop the input_ids
        # because we can't pass both to the underlying model.
        # TODO (ian): Add a warning here?
        if "inputs_embeds" in inputs and "input_ids" in inputs:
            warnings.warn(
                "Both 'inputs_embeds' and 'input_ids' are present in the inputs."
                " Dropping 'input_ids'. (This is fine if the intention was to"
                " pass 'input_ids' in case we were using caching.)"
            )
            inputs.pop("input_ids")

        # Accelerator will *usually* handle moving things to the right device,
        # but occasionally we will run without a prepared DataLoader (e.g. in
        # get_caching_model_with_example), so we need to handle that case.
        # TODO (ian): Work out where to move things to the right device.
        inputs = dict_to_device(inputs, self.model.device)
        return self.model.forward(**inputs)

    def _to_transformers_generation_config(
        self, gen_config: GenerationConfig
    ) -> transformers.GenerationConfig:
        """Converts our GenerationConfig to a transformers GenerationConfig.

        This is necessary because transformers expects its own GenerationConfig object.
        """
        gen_config_dict = dataclasses.asdict(gen_config)
        # Add eos_token_id and pad_token_id to the config.
        gen_config_dict["eos_token_id"] = self.right_tokenizer.eos_token_id
        gen_config_dict["pad_token_id"] = self.right_tokenizer.pad_token_id
        return transformers.GenerationConfig.from_dict(gen_config_dict)

    def generate(self, **inputs):
        """Generate text.

        This wrapper is mostly here in case it's needed for compatibility with
        CachingWrappedModel.
        """
        if inputs.get("generation_config") is not None:
            gen_config = inputs["generation_config"]
            if isinstance(gen_config, GenerationConfig):
                gen_config = self._to_transformers_generation_config(gen_config)
            inputs["generation_config"] = gen_config
        elif self.generation_config is not None:
            gen_config = self._to_transformers_generation_config(self.generation_config)
            inputs["generation_config"] = gen_config
        if inputs.get("tokenizer") is None:
            # Need to specify tokenizer due to stop strings.
            # We use left_tokenizer because we always use left padding for generation.
            inputs["tokenizer"] = self.left_tokenizer

        # Hack to make sure FSDP all-gathers the parameters before .generate. See:
        # https://github.com/pytorch/pytorch/issues/100069
        # TODO(GH#512): Find a way to avoid an additional forward pass/gather.
        if isinstance(self.model, FSDP):
            with torch.no_grad():
                self.model.forward(input_ids=inputs["input_ids"])
        with FSDP.summon_full_params(self.model, recurse=False):
            return self.model.generate(
                **inputs,
                # synced_gpus prevents sporadic hanging when using FSDP, which
                # comes from different generated sequence lengths on different
                # GPUs. We only want to use synced_gpus when we're actually
                # using accelerate, otherwise it throws an error.
                synced_gpus=torch.distributed.is_initialized(),
            )

    def to(self, device: torch.device) -> WrappedModel:
        """Move the model to the given device.

        This moves the underlying model to the given device,
        like the cpu or a cuda gpu.

        TODO (ian): Remove this method when we stop using pipelines.
        """
        # For some reason, the type hint for to() is wrong in transformers
        self.model.to(device=device)  # type: ignore[reportCallIssue]
        return self

    @property
    def config(self) -> PretrainedConfig:
        """Return's the model's config.

        NOTE: This is NOT our ModelConfig object:
        this is the huggingface transformers.PretrainedConfig.
        """
        return self.model.config

    def eval(self) -> WrappedModel:
        """Sets the model to evaluation mode."""
        self.model.eval()
        return self

    def train(self) -> WrappedModel:
        """Sets the model to training mode."""
        self.model.train()
        return self

    @classmethod
    @abstractmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> PreTrainedTokenizerBase:
        pass

    @cached_property
    def left_tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizer to use for left padding."""
        self._left_tokenizer = copy.deepcopy(self.right_tokenizer)
        self._left_tokenizer.padding_side = "left"
        return self._left_tokenizer

    def tokenize(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        padding_side: str | None = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """Tokenize the input text, optionally applying the chat template.

        For now, we assume that the input text is a single 'user' message.

        Args:
            text:
                The input text.
            return_tensors:
                Whether to return tensors, and what type of tensors to return.
            padding_side:
                Whether to pad the input. None means no padding, "right" means
                do right padding, "left" means do left padding.
            add_special_tokens:
                Whether to add special tokens when tokenizing.
            kwargs:
                Included for compatibility with subclasses that have additional
                arguments. In particular, WrappedChatModel has arguments related
                to the chat template.

        Returns:
            The tokenized input, as you'd get from calling a tokenizer.

        """
        if padding_side is None:
            # With no padding, we can use either tokenizer. We already loaded
            # the right-padding one so we just use that.
            return self.right_tokenizer(
                text=text,
                return_tensors=return_tensors,
                padding=False,
                add_special_tokens=add_special_tokens,
            )
        elif padding_side == "right":
            return self.right_tokenizer(
                text=text,
                return_tensors=return_tensors,
                padding=True,
                add_special_tokens=add_special_tokens,
            )
        elif padding_side == "left":
            return self.left_tokenizer(
                text=text,
                return_tensors=return_tensors,
                padding=True,
                add_special_tokens=add_special_tokens,
            )
        else:
            raise ValueError(f"Unknown padding_side value: {padding_side}")

    def decode(
        self,
        token_ids: torch.Tensor | list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes the token ids into a list of strings.

        This is a wrapper around tokenizer.batch_decode so that we can make the
        tokenizer private.
        """
        # For batch_decode it doesn't matter which padding side we use so we use
        # the right tokenizer which is always loaded.
        return self.right_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor | list[torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decodes the token ids into a list of strings.

        This is a wrapper around tokenizer.batch_decode so that we can make the
        tokenizer private.
        """
        # For batch_decode it doesn't matter which padding side we use so we use
        # the right tokenizer which is always loaded.
        return self.right_tokenizer.batch_decode(
            token_ids,  # type: ignore  # batch_decode actually accepts list of tensors.
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def decode_and_unpad(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = False
    ) -> list[str]:
        """Combines batch_decode and remove_padding_tokens"""
        decoded = self.batch_decode(token_ids, skip_special_tokens)
        return remove_padding_tokens(self.right_tokenizer, decoded)

    def pad(
        self,
        encoded_inputs: dict[str, list[list[int]]],
        padding_side: str,
        return_tensors: str | None = None,
    ) -> BatchEncoding:
        """Pads the input text on the given side."""
        if padding_side == "right":
            tokenizer = self.right_tokenizer
        elif padding_side == "left":
            tokenizer = self.left_tokenizer
        else:
            raise ValueError(
                f"Padding side should be 'left' or 'right', got: {padding_side}"
            )
        return tokenizer.pad(
            encoded_inputs=encoded_inputs,
            return_tensors=return_tensors,
        )

    @property
    def all_special_ids(self) -> list[int]:
        """Returns all special token ids."""
        # Right and left tokenizers should have the same special ids.
        return self.right_tokenizer.all_special_ids

    def decode_tokens(
        self,
        inp: torch.Tensor,
        skip_special_tokens: bool = True,
        try_squeeze: bool = True,
    ) -> str:
        """Decodes the token ids into a string.

        TODO(ian): Remove this function; not much better than tokenizer.batch_decode.
        """
        if len(inp.shape) == 2 and inp.shape[0] == 1 and try_squeeze:
            inp = inp.squeeze()

        strings = self.right_tokenizer.decode(
            inp,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        return strings

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings for the given token ids.

        NOTE: We won't be able to backprop through the returned embeddings
        because we need to detach them when using FSDP.
        """
        token_ids = token_ids.to(self.model.device)
        self._check_for_padding_tokens(token_ids)

        # Doing this inside no_grad is crucial because otherwise we'll try
        # to backprop through the embedding weights after they've been
        # re-sharded.
        with torch.no_grad():
            with FSDP.summon_full_params(self.model, recurse=False):
                return self.model.get_input_embeddings()(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        """Returns a copy of the embedding weights for the model."""
        if self.accelerator is None:
            raise ValueError("An accelerator must be added to the model.")
        # Embedding parameters should be in the top-level FSDP module, so we can
        # get them without recursing.
        with FSDP.summon_full_params(self.model, recurse=False):
            embeddings = self.model.get_input_embeddings()
            # The clone() here is very important so we don't try to access
            # re-sharded weights.
            return embeddings.weight.detach().clone()

    def _check_for_padding_tokens(self, token_ids: torch.Tensor) -> None:
        """Checks if padding tokens are present in the token ids.

        When using inputs_embeds, it's important that there are no padding tokens,
        since they are not handled properly."""
        if self.right_tokenizer.pad_token_id is not None:
            assert (
                self.right_tokenizer.pad_token_id not in token_ids
            ), "Padding tokens are present in the token ids."

    @property
    def vocab_size(self) -> int:
        return self.right_tokenizer.vocab_size  # type: ignore

    @property
    def device(self) -> torch.device:
        return self.model.device

    def get_prompt_template(
        self,
        unmodifiable_prefix: str = "",
        modifiable_infix: str = "",
        unmodifiable_suffix: str = "",
    ) -> PromptTemplate:
        """Returns a PromptTemplate for the given text chunks."""
        return self.prompt_builder.get_prompt_template(
            unmodifiable_prefix,
            modifiable_infix,
            unmodifiable_suffix,
            system_prompt=self.system_prompt,
        )

    @overload
    def maybe_apply_chat_template(self, text: str) -> str: ...

    @overload
    def maybe_apply_chat_template(self, text: list[str]) -> list[str]: ...

    def maybe_apply_chat_template(self, text: str | list[str]) -> str | list[str]:
        """If working with a chat model, return text with chat template applied.

        Since this is the base class, we just return the text as is.

        Args:
            text: The text to apply the chat template to.

        Returns:
            The text with the chat template applied.
        """
        return text

    @staticmethod
    def set_seed(seed: int) -> None:
        """Wrapper around transformers set_seed."""
        set_seed(seed % (2**32))

    @property
    def prompt_builder(self) -> PromptTemplateBuilder:
        return PromptTemplateBuilder(
            prompt_prefix="",
            system_prefix="",
            system_suffix="",
            user_prefix="",
            user_suffix="",
        )
