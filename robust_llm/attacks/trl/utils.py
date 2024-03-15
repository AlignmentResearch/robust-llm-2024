from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import torch
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextClassificationPipeline,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from typing_extensions import override

from robust_llm.configs import AttackConfig
from robust_llm.model_utils import _prepare_tokenizer


def make_ppo_trainer(
    attack_config: AttackConfig,
    adversary_model: PreTrainedModel,
    adversary_tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
) -> PPOTrainer:
    ppo_config = {
        "exp_name": "rl-adversary",
        "batch_size": attack_config.trl_attack_config.batch_size,
        "mini_batch_size": attack_config.trl_attack_config.mini_batch_size,
        "gradient_accumulation_steps": attack_config.trl_attack_config.gradient_accumulation_steps,  # noqa: E501
        "seed": attack_config.seed,
        "ppo_epochs": attack_config.trl_attack_config.ppo_epochs,
        # Needed in order to not delete "text_chunked"
        "remove_unused_columns": False,
        # TODO(niki): log locally too?
        "log_with": "wandb",
    }
    config = PPOConfig(**ppo_config)  # type: ignore
    print("the RLAdversary config is", config)
    print("the dataset is", dataset)
    print("dataset size", len(dataset["text"]))
    ppo_trainer = PPOTrainer(
        config=config,
        model=adversary_model,  # type: ignore
        tokenizer=adversary_tokenizer,
        dataset=dataset,
        # Needed to properly process "text_chunked"
        data_collator=trl_data_collator,
    )
    assert isinstance(ppo_trainer, PPOTrainer)
    return ppo_trainer


class LogitTextClassificationPipeline(TextClassificationPipeline):
    @override
    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return model_outputs["logits"]


def trl_data_collator(datapoints: Sequence[Any]) -> Mapping[str, Any]:
    """A simple data collator to use when the dataset contains "text_chunked".

    The default PPOTrainer data collator does not support
    collating lists of lists cleanly, so it is necessary to either
    use our own or modify the output of the default data collator
    after the fact.
    """
    assert all("label" in datapoint for datapoint in datapoints)
    assert all("text" in datapoint for datapoint in datapoints)
    assert all("text_chunked" in datapoint for datapoint in datapoints)

    batch = {}
    batch["label"] = [datapoint["label"] for datapoint in datapoints]
    batch["text"] = [datapoint["text"] for datapoint in datapoints]
    batch["text_chunked"] = [datapoint["text_chunked"] for datapoint in datapoints]
    return batch


def prepare_adversary_model_and_tokenizer(
    attack_config: AttackConfig,
    device: torch.device,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    base_model_name = attack_config.trl_attack_config.adversary_base_model_name
    if "pythia" not in base_model_name:
        raise NotImplementedError(
            "Only Pythia models are currently supported for RL adversaries."
        )

    checkpoint = attack_config.trl_attack_config.adversary_base_model_checkpoint
    checkpoint_string = f"step{checkpoint}"

    adversary_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        revision=checkpoint_string,
        use_cache=False,  # otherwise returns last key/values attentions
        num_labels=2,
    ).to(device)
    adversary_model.config.pad_token_id = adversary_model.config.eos_token_id

    adversary_tokenizer = _prepare_tokenizer(
        model_name_or_path=base_model_name,
        is_pythia=True,
        checkpoint=checkpoint,
        padding_side="left",
    )

    return adversary_model, adversary_tokenizer


def prepare_prompts(
    text_chunked: Sequence[Sequence[str]],
    modifiable_chunks_spec: Sequence[bool],
    response_text: str | Sequence[str],
    append_to_modifiable_chunk: bool = False,
) -> Sequence[str]:
    """Prepare prompts either for the adversary model or for the victim model.

    For a given sequence of chunks of text, works by concatenating all the chunks
    and replacing the modifiable chunk with the response text. If a single response
    is provided, it is used for all the chunks. If a sequence of responses is
    provided, each response is used for the corresponding chunk.

    Args:
        text_chunked: Sequence of chunked texts. Each chunked text is
            a sequence of strings, one of which is modifiable, the others
            of which are not, as determined by modifiable_chunks_spec.
        modifiable_chunks_spec: Specification for which chunks of the original text can
            be modified.
        response_text:
            The text or sequence of texts to replace the modifiable chunks with.
        append_to_modifiable_chunk: if False, the modifiable chunk is replaced by the
            response text. Otherwise, response text is added after the original
            content of the modifiable chunk.

    Returns:
        A sequence of prompts with the modifiable chunks replaced by the
        response text.
    """
    one_response = True
    if not isinstance(response_text, str):
        assert len(response_text) == len(text_chunked)
        one_response = False

    contexts = []
    for i, line in enumerate(text_chunked):
        context_list = []
        for text, is_modifiable in zip(line, modifiable_chunks_spec):
            if is_modifiable:
                replacement_text = response_text if one_response else response_text[i]
                assert isinstance(replacement_text, str)
                if append_to_modifiable_chunk:
                    context_list.append(text)
                context_list.append(replacement_text)
            else:
                context_list.append(text)
        contexts.append("".join(context_list))

    return contexts
