from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar, Union, overload

import torch
import torch.distributed
import torch.distributed as dist
import transformers
from accelerate import Accelerator
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers import (
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.modeling_outputs import ModelOutput

from robust_llm import logger
from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.dist_utils import (
    DistributedRNG,
    broadcast_int,
    broadcast_int128,
    is_main_process,
    rmtree_if_exists,
    try_except_main_process_loop,
)
from robust_llm.logging_utils import log
from robust_llm.models.model_disk_utils import (
    get_model_load_path,
    mark_model_save_as_finished,
)
from robust_llm.models.model_utils import (
    AutoregressiveOutput,
    InferenceType,
    SuppressPadTokenWarning,
    build_dataloader,
    compute_batch_sizes_from_config,
    dict_to_device,
    load_hf_model,
    maybe_no_grad,
    prepare_model_with_accelerate,
    remove_padding_tokens,
)
from robust_llm.models.prompt_templates import (
    AttackChunks,
    Conversation,
    PromptTemplate,
)
from robust_llm.utils import is_correctly_padded, print_time

Prompt = TypeVar("Prompt", str, list[str])


class FlopCount:

    def __init__(self, flops, forward_calls, backward_calls):
        self.start_flops = flops
        self.end_flops = flops
        self.flops = 0
        self.start_forward_calls = forward_calls
        self.end_forward_calls = forward_calls
        self.forward_calls = 0
        self.start_backward_calls = backward_calls
        self.end_backward_calls = backward_calls
        self.backward_calls = 0

    def update(self, flops: int, forward_calls: int, backward_calls: int):
        self.end_flops = flops
        self.flops = self.end_flops - self.start_flops
        self.end_forward_calls = forward_calls
        self.forward_calls = self.end_forward_calls - self.start_forward_calls
        self.end_backward_calls = backward_calls
        self.backward_calls = self.end_backward_calls - self.start_backward_calls


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
        effective_batch_size: int = 8,
        generation_config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize a WrappedModel.

        Args:
            model: The model to wrap.
            right_tokenizer: The tokenizer to use. Should be right-padded (the
                left-padded version will be loaded if needed.)
            accelerator: The accelerator to use.
            inference_type: The type of inference this model is for ('generation'
                or 'classification')
            train_minibatch_size: The minibatch size to use for training.
            eval_minibatch_size: The minibatch size to use for evaluation.
            effective_batch_size: The product of the train batch size and gradient
                accumulation steps. This is useful when we have to use very small
                minibatches due to limited VRAM, but want to simulate a larger batch
                size.
            family: The family of the model, useful for logging model details
                to wandb alongside experiment results.
            generation_config: The generation config to use for generation.
            system_prompt: The system prompt to use for chat models.
                If None, the default system prompt will be used.
            seed: The seed to use for text generation.
        """
        # We need to compute the number of parameters before any accelerate preparation
        # because the model will be sharded across devices.
        # NOTE: If we switch to loading directly to devices, we'll need to change how we
        # compute the number of parameters.
        self._n_params = model.num_parameters()
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
        self.effective_batch_size = effective_batch_size
        self.generation_config = generation_config
        self.system_prompt = system_prompt
        self.seed = seed

        # N.B. to avoid double counting, these are only updated in the main process
        self._flop_count = 0
        self._n_forward_calls = 0
        self._n_backward_calls = 0
        self._input_shapes: list[tuple] = []

        self.register_hooks()
        self._skip_hooks = False

        self.post_init()

    @property
    def gradient_accumulation_steps(self) -> int:
        return max(
            1,
            self.effective_batch_size
            // (self.train_minibatch_size * self.num_processes),
        )

    @cached_property
    def num_processes(self) -> int:
        return (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

    def get_minibatch_size(
        self, input_ids: torch.Tensor, batch_size: int | None
    ) -> int:
        """Returns the minibatch size to use for the given input_ids."""
        batch_size = batch_size or self.eval_minibatch_size
        batch_dim = input_ids.shape[0]
        # Ensure mb_size>=1 even if there are more batches than processes
        return min(batch_size, max(1, batch_dim // self.num_processes))

    def register_hooks(self):
        """Registers hooks to track FLOPs during forward and backward passes.

        We have to loop through the modules to find the right module to register
        the forward hook where we can access the input IDs. We can't just register
        the forward hook on the model itself because then the input is empty.
        """
        module = None
        forward_done = False
        for module in self.model.modules():
            if isinstance(module, FSDP) or isinstance(
                module, transformers.PreTrainedModel
            ):
                continue
            if not forward_done:
                module.register_forward_hook(self.forward_hook)
                forward_done = True
        assert module is not None
        assert forward_done, "Could not find a module to register a forward hook on."
        module.register_full_backward_hook(self.backward_hook)

    def post_init(self) -> None:
        """Runs any model-specific set-up required.

        We split this out from __init__ to avoid having to subclass __init__
        every time, which results in having a lot of boilerplate code.
        """

    def forward_hook(self, module, inputs, outputs):
        """Hook to track FLOPs during forward pass."""
        if self._skip_hooks:
            return
        if self.accelerator is not None:
            # We must gather_for_metrics on all processes to remove dummy inputs
            # due to batching/wrapping and to avoid hanging.
            inputs = self.accelerator.gather_for_metrics(inputs)
            if not is_main_process():
                return
        self._n_forward_calls += 1
        self._input_shapes.append(inputs[0].shape)
        self._input_dict = {"input_ids": inputs[0]}
        self.update_flop_count(
            input_dict=self._input_dict,
            backward=False,
        )

    def backward_hook(self, module, grad_input, grad_output):
        """Hook to track FLOPs during backward pass."""
        if self._skip_hooks:
            return
        if not is_main_process():
            return
        self._n_backward_calls += 1
        self.update_flop_count(
            input_dict=self._input_dict,
            backward=True,
        )

    def compute_flops(
        self,
        input_dict: dict[str, Union[torch.Tensor, Any]],
        backward: bool = False,
    ) -> int:
        """Estimate FLOPs to forward/backward pass through the model.

        Based on PreTrainedModel.floating_point_ops().
        Apparently comes from this paper: https://arxiv.org/pdf/2001.08361.pdf
        """
        flops = (
            (4 if backward else 2)
            * self.model.estimate_tokens(input_dict)
            * self.n_params
        )
        return flops

    def update_flop_count(
        self,
        input_dict: dict[str, Union[torch.Tensor, Any]],
        backward: bool = False,
    ):
        """Update the FLOP count based on the input.

        N.B. should gather_for_metrics before calling this function.
        """
        assert is_main_process()
        self._flop_count += self.compute_flops(
            input_dict=input_dict,
            backward=backward,
        )

    @property
    def n_params(self) -> int:
        return self._n_params

    def broadcast_int(self, data: int | None) -> int:
        if self.accelerator is None:
            assert data is not None
            return data
        return broadcast_int(data, self.accelerator)

    def broadcast_int128(self, data: int | None) -> int:
        if self.accelerator is None:
            assert data is not None
            return data
        return broadcast_int128(data, self.accelerator)

    @contextmanager
    def flop_count_context(self):
        """Main entrypoint for tracking FLOPs, broadcasting to all processes.

        Use `with model.flop_count_context() as flop_count`, then access
        `flop_count.flops` after the `with` block.
        """
        out = None  # Initialize out with a default value
        try:
            out = FlopCount(
                flops=self.broadcast_int128(self._flop_count),
                forward_calls=self.broadcast_int(self._n_forward_calls),
                backward_calls=self.broadcast_int(self._n_backward_calls),
            )
            yield out
        finally:
            if out is not None:  # Check if out was successfully created
                out.update(
                    flops=self.broadcast_int128(self._flop_count),
                    forward_calls=self.broadcast_int(self._n_forward_calls),
                    backward_calls=self.broadcast_int(self._n_backward_calls),
                )

    @contextmanager
    def dont_count_flops(self):
        """Context manager to temporarily disable FLOP counting."""
        try:
            self._skip_hooks = True
            yield
        finally:
            self._skip_hooks = False

    @print_time()
    def push_to_hub(
        self,
        repo_id: str,
        revision: Optional[str],
        retries: int,
        cooldown_seconds: float,
        local_dir: Path | None = None,
    ):
        """Pushes the model and tokenizer to the hub with retries."""
        save_dir = local_dir or Path(tempfile.mkdtemp())
        self.save_local(save_dir, retries=retries, cooldown_seconds=cooldown_seconds)

        assert self.accelerator is not None
        try_except_main_process_loop(
            retries=retries,
            cooldown_seconds=cooldown_seconds,
            accelerator=self.accelerator,
            func=self._push_to_hub_once,
            repo_id=repo_id,
            revision=revision,
            save_dir=save_dir,
            # cleanup files if not using a permanent directory
            cleanup=local_dir is None,
        )
        log("DEBUG: Pushed model to hub.", main_process_only=False)

    @print_time()
    def save_local(
        self,
        output_dir: Path,
        retries: int,
        cooldown_seconds: float,
    ):
        """Save the model and tokenizer to a local directory"""
        if dist.is_initialized():
            assert self.accelerator is not None
            # Configure FSDP state dict settings
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )

            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config,
            ):
                state_dict = self.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        try_except_main_process_loop(
            retries=retries,
            cooldown_seconds=cooldown_seconds,
            accelerator=self.accelerator,
            func=self._save_local,
            state_dict=state_dict,
            output_dir=output_dir,
        )

    def _save_local(
        self,
        state_dict: dict[str, torch.Tensor],
        output_dir: Path,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save with proper serialization settings
        torch.save(
            state_dict,
            output_dir / "pytorch_model.bin",
        )

        # Save other files
        self.model.config.save_pretrained(output_dir)
        self.right_tokenizer.save_pretrained(
            save_directory=output_dir,
        )

        # Create index file
        with open(output_dir / "pytorch_model.bin.index.json", "w") as f:
            json.dump(
                {
                    "metadata": {"format": "pt"},
                    "weight_map": {"": "pytorch_model.bin"},
                },
                f,
            )
        mark_model_save_as_finished(model_save_directory=output_dir)

    @print_time()
    def _push_to_hub_once(
        self,
        repo_id: str,
        revision: Optional[str],
        save_dir: Path,
        cleanup: bool = True,
    ):
        """Makes one attempt to push the model and tokenizer to the hub.

        Args:
            repo_id: The ID of the huggingface hub repo to push to.
            revision: The revision to push to.
            save_dir: The directory to save the model and tokenizer to.
            cleanup: If this is True, we will delete the save_dir
        """
        # Now separately push to hub, using an internal transformers
        # function. We have to use an internal function because both
        # `push_to_hub` and `save_pretrained` do not give enough flexibility
        # when using FSDP: `push_to_hub` doesn't let you pass in a
        # state_dict and `save_pretrained` doesn't let you pass in a
        # tag/revision for the repo.
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
        if cleanup:
            logger.debug(f"Removing temp directory: {save_dir}")
            rmtree_if_exists(save_dir)

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
    @print_time()
    def from_config(
        cls,
        config: ModelConfig,
        accelerator: Accelerator | None,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> WrappedModel:
        """Wrapper around _from_config to try different model load locations.

        Raises:
            OSError: If the model cannot be loaded from any location.
        """
        error_list = []
        try:
            return cls._from_config(
                config,
                accelerator,
                num_classes=num_classes,
                **kwargs,
            )
        except OSError as e:
            error_list.append(e)

        # note(ian): I accidentally uploaded some models as AlignmentResearch/clf_[...]
        # instead of AlignmentResearch/robust_llm_clf_[...]. Thus to avoid keeping
        # track of which is which, we try both and return the first one that
        # works.
        new_name = None
        if config.name_or_path.startswith("AlignmentResearch/robust_llm_"):
            new_name = config.name_or_path.replace(
                "AlignmentResearch/robust_llm_",
                "AlignmentResearch/",
            )
        elif config.name_or_path.startswith("AlignmentResearch/"):
            new_name = config.name_or_path.replace(
                "AlignmentResearch/",
                "AlignmentResearch/robust_llm_",
            )
        if new_name is not None:
            new_config = dataclasses.replace(config, name_or_path=new_name)
            try:
                return cls._from_config(
                    new_config,
                    accelerator,
                    num_classes=num_classes,
                    **kwargs,
                )
            except OSError as e:
                error_list.append(e)
        logger.warning("Failed to load model from HFHub.")

        # HF Hub failed. Let's try loading from disk in case we saved the model
        # locally but have not yet uploaded it.
        model_name = config.name_or_path.removeprefix(
            "AlignmentResearch/robust_llm_"
        ).removeprefix("AlignmentResearch/")
        disk_path = get_model_load_path(
            models_path=Path(config.load_prefix) / "models",
            model_name=model_name,
            revision=config.revision,
        )
        if disk_path is None:
            logger.info("Model not found on disk.")
        else:
            logger.info("Loading model from disk instead: %s", disk_path)
            new_config = dataclasses.replace(config, name_or_path=str(disk_path))
            try:
                return cls._from_config(
                    new_config,
                    accelerator,
                    num_classes=num_classes,
                    **kwargs,
                )
            except OSError as e:
                error_list.append(e)

        logger.error("Failed to load model from HFHub and disk: %s", error_list)
        raise error_list[0]

    @classmethod
    def _from_config(
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
            attention_implementation=config.attention_implementation,
            num_classes=num_classes,
        )

        try:
            subcls = cls._registry[config.family]
        except KeyError:
            raise ValueError(f"Unsupported model family: {config.family}")

        train_mb_size, eval_mb_size = compute_batch_sizes_from_config(config)
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
            effective_batch_size=config.effective_batch_size,
            generation_config=config.generation_config,
            family=config.family,
            system_prompt=config.system_prompt,
            seed=config.seed,
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

        minibatch_size = self.get_minibatch_size(input_ids, minibatch_size)

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

        minibatch_size = self.get_minibatch_size(input_ids, minibatch_size)

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

        minibatch_size = self.get_minibatch_size(input_ids, minibatch_size)

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

    def generate_from_text(self, text: str) -> str:
        """Returns the autoregressive generation from text with some post-processing.

        - Removes the input text from the output text.
        - Removes special tokens from the output text.
        - Removes any stop strings from the end of the output text.
        """
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
        )
        assert isinstance(all_tokens, torch.Tensor)
        assert all_tokens.shape[0] == 1

        # Only keep the newly generated tokens
        output_tokens = all_tokens[0, input_ids.shape[1] :]

        output_text = self.decode(output_tokens, skip_special_tokens=True)

        return output_text

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
        return self.model(**inputs)

    def _to_transformers_generation_config(
        self, gen_config: GenerationConfig
    ) -> transformers.GenerationConfig:
        """Converts our GenerationConfig to a transformers GenerationConfig.

        This is necessary because transformers expects its own GenerationConfig object.
        """
        gen_config_dict = dataclasses.asdict(gen_config)
        # Add eos_token_id and pad_token_id to the config.
        gen_config_dict["eos_token_id"] = self.config.eos_token_id
        gen_config_dict["pad_token_id"] = self.config.pad_token_id
        return transformers.GenerationConfig.from_dict(gen_config_dict)

    @cached_property
    def transformers_generation_config(self) -> transformers.GenerationConfig:
        """Returns a transformers GenerationConfig object."""
        assert self.generation_config is not None
        return self._to_transformers_generation_config(self.generation_config)

    def pad_and_concatenate(self, tensors):
        """Wrapper around torch.cat that pads the tensors to the same length."""
        assert self.right_tokenizer.pad_token_id is not None
        max_length = max(tensor.size(1) for tensor in tensors)
        padded_tensors = []

        for tensor in tensors:
            padding = torch.full(
                (tensor.size(0), max_length - tensor.size(1)),
                self.right_tokenizer.pad_token_id,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded_tensor = torch.cat([tensor, padding], dim=1)
            padded_tensors.append(padded_tensor)

        return torch.cat(padded_tensors, dim=0)

    def _generate_single_call(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: transformers.GenerationConfig,
    ) -> torch.Tensor:
        """Call model.generate once (N.B. params must be all-gathered first)."""
        is_deterministic = torch.are_deterministic_algorithms_enabled()
        if is_deterministic:
            # HF's generation does not have a deterministic implementation.
            torch.use_deterministic_algorithms(False)
        self._set_seed()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.config.pad_token_id,
            generation_config=generation_config,
            tokenizer=self.left_tokenizer,
            # synced_gpus prevents sporadic hanging when using FSDP, which
            # comes from different generated sequence lengths on different
            # GPUs. We only want to use synced_gpus when we're actually
            # using accelerate, otherwise it throws an error.
            synced_gpus=torch.distributed.is_initialized(),
        )
        assert isinstance(out, torch.Tensor)
        if is_deterministic:
            torch.use_deterministic_algorithms(True)
        return out

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: transformers.GenerationConfig | None = None,
    ) -> torch.Tensor:
        """Generate text, being careful to set the seed for every batch."""
        if generation_config is None:
            generation_config = self.transformers_generation_config
        # Hack to make sure FSDP all-gathers the parameters before .generate. See:
        # https://github.com/pytorch/pytorch/issues/100069
        # TODO(GH#512): Find a way to avoid an additional forward pass/gather.
        if isinstance(self.model, FSDP):
            with torch.no_grad():
                self.model.forward(input_ids=input_ids)
        with FSDP.summon_full_params(self.model, recurse=False):
            if generation_config.do_sample:
                # We need to generate one row at a time because we need to set the seed
                # for each row.
                return self.pad_and_concatenate(
                    [
                        self._generate_single_call(row_ids, row_mask, generation_config)
                        for row_ids, row_mask in zip(input_ids, attention_mask)
                    ],
                )
            else:
                return self._generate_single_call(
                    input_ids, attention_mask, generation_config
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

    @overload
    def get_tokens(
        self,
        inputs: str | list[str],
        return_tensors: Literal[None] = None,
        add_special_and_chat: bool = False,
    ) -> list[list[int]]: ...

    @overload
    def get_tokens(
        self,
        inputs: str | list[str],
        return_tensors: Literal["pt"],
        add_special_and_chat: bool = False,
    ) -> torch.Tensor: ...

    def get_tokens(
        self,
        inputs: str | list[str],
        return_tensors: Literal[None, "pt"] = None,
        add_special_and_chat: bool = False,
    ) -> list[list[int]] | torch.Tensor:
        """Tokenize the inputs and return the token ids.

        Use tokenizer which is part of the wrapped model. Handle all the arguments we
        have to add to the tokenizer.

        Args:
            inputs: The input text or list of texts to tokenize.
            return_tensors: Whether to return tensors, and what type of tensors to
                return.
            add_special_and_chat: Whether to add special tokens and use chat template.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        encoded = self.tokenize(
            inputs,
            return_tensors=return_tensors,
            add_special_tokens=add_special_and_chat,
            use_chat_template=add_special_and_chat,
        )
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device=self.device)
        return input_ids  # type: ignore  # mypy thinks it's EncodingFast | Any

    def decode(
        self,
        token_ids: torch.Tensor | list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes the token ids into a list of strings.

        This is a wrapper around tokenizer.decode so that we can make the
        tokenizer private.
        """
        # For decode it doesn't matter which padding side we use so we use
        # the right tokenizer which is always loaded.
        return self.right_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor | list[torch.Tensor] | list[list[int]],
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
        since they are not handled properly.
        """
        if self.config.pad_token_id is not None:
            assert (
                self.config.pad_token_id not in token_ids
            ), "Padding tokens are present in the token ids."

    @property
    def vocab_size(self) -> int:
        return self.right_tokenizer.vocab_size  # type: ignore

    @property
    def device(self) -> torch.device:
        return self.model.device

    def chunks_to_prompt_template(
        self,
        chunks: AttackChunks,
        perturb_min: float,
        perturb_max: float,
        rng: DistributedRNG,
    ) -> PromptTemplate:
        """Returns a PromptTemplate for the given text chunks.

        Handles chat formatting and selecting the position for the attack tokens.
        """
        base_template = chunks.get_prompt_template(perturb_min, perturb_max, rng)
        conv = self.init_conversation()
        return conv.wrap_prompt_template(base_template)

    def maybe_apply_user_template(self, text: Prompt) -> Prompt:
        """If working with a chat model, return text with chat template applied.

        Since this is the base class, we just return the text as is.

        Args:
            text: The user prompt(s) to apply the chat template to.

        Returns:
            The text with the chat template applied.
        """
        return text

    def _set_seed(self) -> None:
        """Wrapper around transformers set_seed."""
        set_seed(self.seed % (2**32))

    def init_conversation(self) -> Conversation:
        return Conversation(
            prompt_prefix="",
            system_prefix="",
            system_suffix="",
            user_prefix="",
            user_suffix="",
            assistant_prefix="",
            assistant_suffix="",
        )

    def clean_chat_artifacts(self, text: str) -> str:
        """Cleans up chat artifacts from the text."""
        conv = self.init_conversation()
        return conv.clean_special_strings(text)
