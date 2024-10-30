from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

from robust_llm.config.configs import ExperimentConfig
from robust_llm.dist_utils import DistributedRNG, pad_batch_across_processes
from robust_llm.logging_utils import (
    TRAIN_LOG_INTERVAL_SECS,
    format_training_status,
    log,
    wandb_log,
)
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
from robust_llm.training.training_utils import prepare_dataloader
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
            log("No saved state found. Starting from scratch.", main_process_only=False)

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
    val_dataset = load_rllm_dataset(config.dataset, split="validation")
    val_dataset = val_dataset.tokenize(model_state.wrapped_model.right_tokenizer)

    if training_config.adversarial is not None:
        clean_ds = clean_dataset.for_training()
        adversarial_ds = Dataset.from_dict(
            {f: [] for f in clean_ds.features},
        ).with_format("torch")
    else:
        adversarial_ds = None

    dataset_state = DatasetState(
        clean_dataset=clean_dataset,
        val_dataset=val_dataset,
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

    dataset = state.get_full_dataset()
    dataloader = prepare_dataloader(victim, dataset)

    dataloader, model, optimizer = state.accelerator.prepare(
        dataloader,
        model,
        optimizer,
    )

    # Train for one epoch.
    start_time = time.perf_counter()
    last_log_time = start_time
    n_batches = len(dataloader)
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

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        gathered_loss = state.accelerator.gather(outputs.loss)
        assert isinstance(gathered_loss, torch.Tensor)
        average_loss = gathered_loss.mean()
        wandb_log(
            {
                "train/loss": average_loss.item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/flops": state.flops,
            },
            commit=True,
        )
        # Log if it has been at least TRAIN_LOG_INTERVAL_SECS seconds.
        current_time = time.perf_counter()
        if (
            current_time - last_log_time >= TRAIN_LOG_INTERVAL_SECS
            or i == n_batches - 1  # Always log at the end of the epoch.
        ):
            elapsed = current_time - start_time
            status = format_training_status(
                epoch=state.epoch,
                batch=i + 1,
                total_batches=n_batches,
                loss=average_loss.item(),
                elapsed_secs=elapsed,
            )
            log(status)
            last_log_time = current_time

    # dataloader does have a property called total_dataset_length.
    if len(losses) != dataloader.total_dataset_length:  # type: ignore
        raise ValueError(
            f"Expected {len(losses)=}"
            f" to equal {dataloader.total_dataset_length=}"  # type: ignore
        )

    # Do various evaluations at the end of some epochs.
    if state.should_evaluate():
        state.evaluate(local_files_path=Path(state.config.environment.save_root))

    state.update_after_epoch(losses)
    return state
