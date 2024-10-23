from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

from robust_llm.config.configs import ExperimentConfig
from robust_llm.dist_utils import DistributedRNG, pad_batch_across_processes
from robust_llm.logging_utils import log, wandb_log
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.training.state_classes import (
    OPTIMIZER_MAP,
    AdversarialPipelineState,
    DatasetState,
    ModelState,
    RNGState,
    TrainingPipelineState,
    TrainingState,
    build_lr_scheduler,
)
from robust_llm.utils import print_time

# Avoid spam from map/filter in datasets library.
disable_progress_bar()


def run_train_loop(config: ExperimentConfig, accelerator: Accelerator):
    """Main training loop.

    Args:
        config: The configuration for the training run.
        accelerator: The Accelerator object to use for training.
    """
    assert config.training is not None

    # TODO(ian): Maybe separate resuming and saving
    resume_from_checkpoint = config.environment.allow_checkpointing
    resume_mode = config.environment.resume_mode
    save_checkpoints = config.environment.allow_checkpointing

    base_path = Path(config.environment.save_root)
    checkpoints_path = base_path / "checkpoints"
    models_path = base_path / "models"
    state = get_first_state(
        config,
        accelerator,
        checkpoints_path,
        resume_from_checkpoint,
    )

    while not state.training_is_finished():
        with print_time(f"train_one_epoch (epoch {state.epoch})"):
            state = train_one_epoch(state)

        if save_checkpoints:
            state.save(checkpoints_path)
            state.cleanup_checkpoints(checkpoints_path)

        if state.should_save_trained_model():
            state.save_trained_model(models_path)

        # This is mostly for debugging, since we usually won't want to reload
        # the state every time.
        if resume_from_checkpoint and resume_mode == "always":
            state = get_state_subclass(config).load(
                config=config,
                path=checkpoints_path,
                accelerator=accelerator,
            )


@print_time()
def get_first_state(
    config: ExperimentConfig,
    accelerator: Accelerator,
    base_path: Path,
    resume_from_checkpoint: bool,
) -> TrainingPipelineState:
    """Wrapper around _get_first_state for logging."""
    state = _get_first_state(
        config,
        accelerator,
        base_path,
        resume_from_checkpoint,
    )
    model_info_dict = {
        "model_family": state.model_state.wrapped_model.family,
        "model_size": state.model_state.wrapped_model.n_params,
    }
    wandb_log(model_info_dict, commit=True)
    return state


def _get_first_state(
    config: ExperimentConfig,
    accelerator: Accelerator,
    base_path: Path,
    resume_from_checkpoint: bool,
) -> TrainingPipelineState:
    if resume_from_checkpoint:
        try:
            return get_state_subclass(config).load(
                config=config, path=base_path, accelerator=accelerator
            )
        except FileNotFoundError:
            log("No saved state found. Starting from scratch.")

    return initialize_state(config, accelerator=accelerator)


def initialize_state(
    config: ExperimentConfig, accelerator: Accelerator
) -> TrainingPipelineState:
    training_config = config.training
    assert training_config is not None
    torch.random.manual_seed(training_config.seed)
    model_state = ModelState(
        wrapped_model=WrappedModel.from_config(config.model, accelerator)
    )

    optim_class = OPTIMIZER_MAP[training_config.optimizer]
    # Ignore the type since it doesn't understand partial
    optimizer = optim_class(  # type: ignore
        model_state.wrapped_model.model.parameters(),
        lr=training_config.learning_rate,
    )

    lr_scheduler = build_lr_scheduler(optimizer, config, accelerator)
    training_state = TrainingState(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    rng_state = RNGState(
        torch_rng_state=torch.random.get_rng_state(),
        distributed_rng=DistributedRNG(
            seed=training_config.seed,
            accelerator=accelerator,
        ),
    )

    clean_dataset = load_rllm_dataset(config.dataset, split="train")
    clean_dataset = clean_dataset.tokenize(model_state.wrapped_model.right_tokenizer)
    if training_config.adversarial is not None:
        clean_ds = clean_dataset.for_training()
        adversarial_ds = Dataset.from_dict(
            {f: [] for f in clean_ds.features},
        ).with_format("torch")
    else:
        adversarial_ds = None

    dataset_state = DatasetState(
        clean_dataset=clean_dataset,
        adv_dataset=adversarial_ds,
        # TODO(ian): Work out if we should shuffle the clean dataset at the start.
        # I don't think we need to because the datasets we have on huggingface are
        # already shuffled.
        clean_index_map={i: i for i in range(len(clean_dataset))},
    )

    return get_state_subclass(config)(
        epoch=0,
        accelerator=accelerator,
        config=config,
        dataset_state=dataset_state,
        model_state=model_state,
        training_state=training_state,
        rng_state=rng_state,
        flops=0,
    )


def get_state_subclass(config: ExperimentConfig) -> type[TrainingPipelineState]:
    """Get the state subclass for this experiment."""
    assert config.training is not None
    if config.training.adversarial is not None:
        return AdversarialPipelineState
    return TrainingPipelineState


def train_one_epoch(state: TrainingPipelineState) -> TrainingPipelineState:
    optimizer = state.training_state.optimizer
    lr_scheduler = state.training_state.lr_scheduler
    victim = state.model_state.wrapped_model
    model = victim.model
    optimizer.zero_grad()

    if state.should_augment_dataset():
        state.augment_dataset()

    dataloader = _prepare_dataloader(state)

    dataloader, model, optimizer = state.accelerator.prepare(
        dataloader,
        model,
        optimizer,
    )

    # Train for one epoch.
    # TODO(ian): Work out if I can put these losses somewhere nicer.
    losses: list[torch.Tensor] = []
    for i, batch in enumerate(dataloader):
        batch = pad_batch_across_processes(
            state.model_state.wrapped_model.right_tokenizer,
            state.accelerator,
            batch,
        )
        model.train()
        with state.model_state.wrapped_model.flop_count_context() as forward_flops:
            outputs = model(**batch)
        state.flops += forward_flops.flops
        log(f"Forward flops: {forward_flops.flops:.2E}")

        # TODO(ian): Work out if I can put this loss calc somewhere nicer.
        logits = outputs["logits"]
        labels = batch["labels"]
        logits, labels = state.accelerator.gather_for_metrics((logits, labels))
        batch_losses = F.cross_entropy(logits, labels, reduction="none")

        losses += batch_losses

        # TODO(ian): Work out if we should use a different loss (gathered?).
        with state.model_state.wrapped_model.flop_count_context() as backward_flops:
            state.accelerator.backward(outputs.loss)
        state.flops += backward_flops.flops
        log(f"Backward flops: {backward_flops.flops:.2E}")

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        gathered_loss = state.accelerator.gather(outputs.loss)
        assert isinstance(gathered_loss, torch.Tensor)
        average_loss = gathered_loss.mean()
        wandb_log(
            {
                "loss": average_loss.item(),
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "flops": state.flops,
            },
            commit=True,
        )
        log(
            f"Epoch {state.epoch}, batch {i}: "
            f"Mean loss across processes: {average_loss.item()}"
        )

    # dataloader does have a property called total_dataset_length.
    if len(losses) != dataloader.total_dataset_length:  # type: ignore
        raise ValueError(
            f"Expected {len(losses)=}"
            f" to equal {dataloader.total_dataset_length=}"  # type: ignore
        )
    state.update_after_epoch(losses)
    return state


def _prepare_dataloader(state: TrainingPipelineState) -> DataLoader:
    ds = state.get_full_dataset()
    # Computation taken from the old training.py.
    victim = state.model_state.wrapped_model
    batch_size = min(
        len(ds) // victim.num_processes,
        victim.train_minibatch_size,
    )

    tokenizer = victim.right_tokenizer
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    current_columns = ds.column_names
    columns_to_keep = ["input_ids", "label"]
    columns_to_remove = [col for col in current_columns if col not in columns_to_keep]
    ds = ds.map(
        lambda x: {k: v for k, v in x.items() if k in columns_to_keep},
        remove_columns=columns_to_remove,
    )

    dataloader = DataLoader(
        ds,  # type: ignore  # HF datasets can actually be used here.
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    return dataloader
