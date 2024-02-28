from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import torch
import wandb
from accelerate.utils import gather_object
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
) -> PPOTrainerWithModifiedLogging:
    ppo_config = {
        "exp_name": "rl-adversary",
        "batch_size": attack_config.trl_attack_config.batch_size,
        "mini_batch_size": attack_config.trl_attack_config.mini_batch_size,
        "gradient_accumulation_steps": attack_config.trl_attack_config.gradient_accumulation_steps,  # noqa: E501
        "init_kl_coef": attack_config.trl_attack_config.initial_kl_coefficient,
        "seed": attack_config.seed,
        # TODO(niki): try not False
        "adap_kl_ctrl": False,
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
    ppo_trainer = PPOTrainerWithModifiedLogging(
        config=config,
        model=adversary_model,  # type: ignore
        tokenizer=adversary_tokenizer,
        dataset=dataset,
        # Needed to properly process "text_chunked"
        data_collator=trl_data_collator,
    )
    assert isinstance(ppo_trainer, PPOTrainerWithModifiedLogging)
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


class PPOTrainerWithModifiedLogging(PPOTrainer):
    """A PPOTrainer with customized log_stats for our use case.

    This code is heavily based on that of the original PPOTrainer,
    found at https://github.com/huggingface/trl/blob/1bfe0b8fcb02d91d842cdc64e8810871d2d5fd91/trl/trainer/ppo_trainer.py#L109  # noqa: E501
    """

    @override
    def log_stats(  # type: ignore
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: List[str],
    ) -> None:
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
            columns_to_log (`List[str]`):
                Which columns to log from the `batch` dictionary. Behaves slightly
                differently from original PPOTrainer.log_stats in that it does not
                require the batch to contain "query" and "response" keys.
        """

        prepend_string = "trl_training/"

        # all gather stats
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.current_device)  # type: ignore
        rewards = self.accelerator.gather(rewards).flatten()  # type: ignore

        if self.config.log_with == "wandb":  # type: ignore
            if any(
                [column_to_log not in batch.keys() for column_to_log in columns_to_log]
            ):
                raise ValueError(
                    f"Columns to log {columns_to_log} are not "
                    "present in the batch {batch.keys()}."
                )

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
            if self.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b)
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            # NOTE(niki): previously, this required the batch to contain
            # "query" and "response" keys. I've removed that requirement
            # because does not seem necessary or helpful for our use case.
            if self.config.log_with == "wandb":  # type: ignore
                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]  # type: ignore # noqa: E501
                logs.update(
                    {
                        "game_log": wandb.Table(
                            columns=[*columns_to_log, "reward"], rows=table_rows
                        )
                    }
                )

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()  # type: ignore # noqa: E501
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()  # type: ignore # noqa: E501
            logs["env/reward_dist"] = rewards.cpu().numpy()  # type: ignore

            # NOTE(niki): I added this prepend step to make it easier
            # to tell which logs are from the TRL training.
            prepended_logs = {prepend_string + k: v for k, v in logs.items()}

            self.accelerator.log(
                prepended_logs,
            )


def prepare_prompts(
    text_chunked: Sequence[Sequence[str]],
    modifiable_chunks_spec: Sequence[bool],
    response_text: str | Sequence[str],
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
        response_text:
            The text or sequence of texts to replace the modifiable chunks with.

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
                context_list.append(replacement_text)
            else:
                context_list.append(text)
        contexts.append("".join(context_list))

    return contexts
