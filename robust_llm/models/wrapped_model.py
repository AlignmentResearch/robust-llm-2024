from __future__ import annotations

import dataclasses
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional

import torch
import transformers
from accelerate import Accelerator
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.modeling_outputs import ModelOutput

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import (
    InferenceType,
    SuppressPadTokenWarning,
    _get_embedding_weights,
    build_dataloader,
    dict_to_device,
    load_hf_model,
    maybe_no_grad,
    prepare_model_with_accelerate,
)


class WrappedModel(ABC):
    """Combines a model and a tokenizer."""

    _registry: dict[str, type[WrappedModel]] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        """Initialize a WrappedModel.

        Args:
            model: The model to wrap.
            tokenizer: The tokenizer to use.
            accelerator: The accelerator to use.
            inference_type: The type of inference this model is for ('generation'
                or 'classification' or 'trl')
            train_minibatch_size: The minibatch size to use for training.
            eval_minibatch_size: The minibatch size to use for evaluation.
            generation_config: The generation config to use for generation.
        """
        self.accelerator = accelerator
        if self.accelerator is not None:
            self.model = prepare_model_with_accelerate(self.accelerator, model)
        else:
            self.model = model

        self.tokenizer = tokenizer
        self.inference_type = inference_type
        self.train_minibatch_size = train_minibatch_size
        self.eval_minibatch_size = eval_minibatch_size
        self.generation_config = generation_config

    def push_to_hub(
        self,
        repo_id: str,
        revision: Optional[str],
        retries: int,
        cooldown_seconds: float,
    ):
        """Pushes the model and tokenizer to the hub with retries."""
        for attempt in range(retries):
            try:
                self.model.push_to_hub(repo_id=repo_id, revision=revision)  # type: ignore # noqa: E501
                # Even though the above line should push both model and tokenizer,
                # in practice the tokenizer sometimes doesn't get pushed,
                # so we do it explicitly here.
                self.tokenizer.push_to_hub(repo_id=repo_id, revision=revision)  # type: ignore # noqa: E501
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
            num_classes=num_classes,
        )

        family = config.family
        # Handle special case where Pythia models are based on GPTNeoX.
        if family == "pythia":
            family = "gpt_neox"
        try:
            subcls = cls._registry[family]
        except KeyError:
            raise ValueError(f"Unsupported model family: {family}")

        tokenizer = subcls.load_tokenizer(config)
        return subcls(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            inference_type=inference_type,
            train_minibatch_size=config.train_minibatch_size,
            eval_minibatch_size=config.eval_minibatch_size,
            generation_config=config.generation_config,
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

        Returns:
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
                minibatch_out = dict_to_device(minibatch_out, "cpu")
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

        Returns:
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
                minibatch_out = dict_to_device(minibatch_out, "cpu")
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

        Returns:
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
    ) -> Iterator[list[str]]:
        """Returns the autoregressive generation from the token ids.

        Args:
            input_ids: The token ids to start the generation from.
            attention_mask: The attention mask for the input_ids. Only needed
                if the input_ids are actually padded.
            minibatch_size: The minibatch size to use. If None, we use
                self.eval_minibatch_size.
        Returns:
            A list of strings, which are the generated sequences.
        """
        assert self.inference_type == InferenceType.GENERATION
        if attention_mask is not None and any(attention_mask[:, -1] == 0):
            raise ValueError("It seems like your inputs are right-padded.")

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
                assert isinstance(minibatch_tokens, torch.Tensor)
                minibatch_texts = self.tokenizer.batch_decode(
                    minibatch_tokens, skip_special_tokens=True
                )
                yield minibatch_texts

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
        gen_config_dict["eos_token_id"] = self.tokenizer.eos_token_id
        gen_config_dict["pad_token_id"] = self.tokenizer.pad_token_id
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
        return self.model.generate(**inputs)

    def to(self, device: torch.device) -> WrappedModel:
        """Move the model to the given device.

        This moves the underlying model to the given device,
        like the cpu or a cuda gpu.

        TODO (ian): Remove this method when we stop using pipelines.
        """
        # For some reason, the type hint for to() is wrong in transformers
        self.model.to(device=device)  # type: ignore[reportCallIssue]
        return self

    def can_generate(self) -> bool:
        """Returns whether the model can generate text.

        This is used by pipelines.

        TODO (ian): Remove this method when we stop using pipelines.
        """
        return (self.tokenizer.padding_side == "left") and (
            self.inference_type == InferenceType.GENERATION
        )

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

    def add_accelerator(self, accelerator: Accelerator) -> None:
        """Adds an accelerator to the model."""
        if self.accelerator is not None:
            raise ValueError("An accelerator has already been added to the model.")
        self.model = prepare_model_with_accelerate(
            accelerator=accelerator,
            model=self.model,
        )
        self.accelerator = accelerator

    def get_tokens(
        self,
        inp: str | list[str],
        return_tensors: Optional[str] = "pt",
        add_special: bool = False,
    ) -> torch.Tensor:
        """Handles all the arguments we have to add to the tokenizer."""
        tokens = self.tokenizer(
            inp,
            return_tensors=return_tensors,
            add_special_tokens=add_special,
        ).input_ids
        if return_tensors == "pt":
            tokens = tokens.to(dtype=torch.int64)
        return tokens

    def decode_tokens(
        self,
        inp: torch.Tensor,
        skip_special_tokens: bool = True,
        try_squeeze: bool = True,
    ) -> str:
        if len(inp.shape) == 2 and inp.shape[0] == 1 and try_squeeze:
            inp = inp.squeeze()

        strings = self.tokenizer.decode(
            inp,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        return strings

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings for the given token ids."""
        token_ids = token_ids.to(self.model.device)
        self._check_for_padding_tokens(token_ids)
        return self.model.get_input_embeddings()(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        """Returns the embedding weights for the model."""
        # TODO: work out if we should be adding positional embeddings
        if self.accelerator is None:
            raise ValueError("An accelerator must be added to the model.")
        return _get_embedding_weights(
            self.accelerator, self.model.get_input_embeddings()
        )

    def _check_for_padding_tokens(self, token_ids: torch.Tensor) -> None:
        """Checks if padding tokens are present in the token ids.

        When using inputs_embeds, it's important that there are no padding tokens,
        since they are not handled properly."""
        if self.tokenizer.pad_token_id is not None:
            assert (
                self.tokenizer.pad_token_id not in token_ids
            ), "Padding tokens are present in the token ids."

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size  # type: ignore

    @property
    def device(self) -> torch.device:
        return self.model.device

    @staticmethod
    def set_seed(seed: int) -> None:
        """Wrapper around transformers set_seed."""
        set_seed(seed % (2**32))
