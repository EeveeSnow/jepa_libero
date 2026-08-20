from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.processor_smolvla import (
    make_smolvla_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from jepa_adapter import JepaAdapter, encode_language
from libero_data import (
    TARGET_TASKS,
    IndexedDataset,
    all_frame_indices,
    episode_frame_indices,
    load_frame_dataset,
    load_temporal_dataset,
    matching_episodes,
    normalize_task,
)

import gc
import os

import psutil


PROCESS = psutil.Process(os.getpid())


def print_memory(tag: str) -> None:
    rss_gb = PROCESS.memory_info().rss / 1024**3
    print(f"[memory] {tag}: process_rss={rss_gb:.2f} GB")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_task_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [normalize_task(value)]

    if isinstance(value, (tuple, list)):
        return [normalize_task(x) for x in value]

    raise TypeError(f"Unsupported task batch type: {type(value)}")


def ensure_chw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [H,W,C], got {image.shape}")

    if image.shape[0] == 3:
        return image

    if image.shape[-1] == 3:
        return image.permute(2, 0, 1)

    raise ValueError(f"Cannot identify RGB channel in shape {image.shape}")


def sample_to_camera_tensor(
    sample: dict[str, Any],
    camera_keys: list[str],
) -> torch.Tensor:
    images = [
        ensure_chw(sample[key]).float()
        for key in camera_keys
    ]
    return torch.stack(images, dim=0)


def build_goal_prototypes(
    *,
    adapter: JepaAdapter,
    plain_dataset,
    task_to_episodes: dict[str, list[int]],
    device: torch.device,
    max_endpoints_per_task: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Prototype = average target-encoder embedding of the last frames
    selected demonstrations.

    Используются только видеокадры. Действия endpoint-кадров здесь не читаются.
    """
    adapter.eval()
    result: dict[str, torch.Tensor] = {}

    camera_keys = list(plain_dataset.meta.camera_keys)

    with torch.no_grad():
        for task, episode_indices in task_to_episodes.items():
            selected = episode_indices[:max_endpoints_per_task]
            embeddings = []

            for episode_index in selected:
                row = plain_dataset.meta.episodes[episode_index]
                frame_index = int(row["dataset_to_index"]) - 1

                sample = plain_dataset[frame_index]
                images = sample_to_camera_tensor(
                    sample,
                    camera_keys,
                ).unsqueeze(0).to(device)

                embedding = adapter.target_encoder(images)
                embeddings.append(embedding.squeeze(0).cpu())

            if not embeddings:
                raise RuntimeError(f"No prototype frames for task {task}")

            result[normalize_task(task)] = torch.stack(embeddings).mean(dim=0)

    return result


def preprocess_window(
    raw_batch: dict[str, Any],
    preprocessor,
    camera_keys: list[str],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[str]]:
    """
    LeRobot temporal window содержит:

        camera: [B, 2, C, H, W]

    Индекс 0 — current frame.
    Индекс 1 — future frame.

    В базовую SmolVLA передаётся только current frame.
    """
    tasks = as_task_list(raw_batch["task"])

    future_images = {
        key: raw_batch[key][:, 1].clone()
        for key in camera_keys
    }

    for key in camera_keys:
        raw_batch[key] = raw_batch[key][:, 0].clone()

    # Эти поля нужны только для temporal padding и не используются моделью.
    for key in list(raw_batch.keys()):
        if key.endswith("_is_pad"):
            raw_batch.pop(key)

    processed = preprocessor(raw_batch)

    return processed, future_images, tasks


def stack_processed_cameras(
    batch: dict[str, Any],
    camera_keys: list[str],
) -> torch.Tensor:
    images = []

    for key in camera_keys:
        value = batch[key]

        if value.ndim == 5:
            value = value[:, -1]

        images.append(value.float())

    return torch.stack(images, dim=1)


def stack_future_cameras(
    future_images: dict[str, torch.Tensor],
    camera_keys: list[str],
    device: torch.device,
) -> torch.Tensor:
    images = []

    for key in camera_keys:
        value = future_images[key].to(device).float()
        value = torch.stack(
            [ensure_chw(x) for x in value],
            dim=0,
        )
        images.append(value)

    return torch.stack(images, dim=1)


def goal_tensor(
    tasks: list[str],
    prototypes: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    missing = [task for task in tasks if task not in prototypes]

    if missing:
        raise KeyError(f"Missing goal prototypes: {missing}")

    return torch.stack(
        [prototypes[task] for task in tasks],
        dim=0,
    ).to(device)


def load_seen_policy(
    checkpoint: str,
    seen_dataset,
    device: str,
):
    cfg = PreTrainedConfig.from_pretrained(
        checkpoint,
        device=device,
    )

    cfg.pretrained_path = Path(checkpoint)
    cfg.device = device

    policy = make_policy(
        cfg,
        ds_meta=seen_dataset.meta,
    )

    policy.eval()

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config,
        dataset_stats=seen_dataset.meta.stats,
    )

    return policy, preprocessor, postprocessor


@torch.no_grad()
def predict_base_action(
    base_policy,
    batch: dict[str, Any],
    step: int,
) -> torch.Tensor:
    base_policy.reset()

    batch_size = batch[OBS_STATE].shape[0]
    device = batch[OBS_STATE].device

    shape = (
        batch_size,
        base_policy.config.chunk_size,
        base_policy.config.max_action_dim,
    )

    # Фиксируем seed для воспроизводимости flow-matching sampling.
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(12345)

    noise = base_policy.model.sample_noise(shape, device)

    base_action = base_policy.predict_action_chunk(
        batch,
        noise=noise,
    )

    torch.random.set_rng_state(generator_state)

    return base_action


def train(args) -> None:
    seed_everything(args.seed)

    device = torch.device(args.device)
    print_memory("startup")
    temporal_seen = load_temporal_dataset(
        args.data_root,
        "libero_90",
        chunk_size=args.chunk_size,
        future_offset=args.future_offset,
        video_backend=args.video_backend,
    )

    plain_seen = load_frame_dataset(
        args.data_root,
        "libero_90",
        video_backend=args.video_backend,
    )

    base_policy, preprocessor, _ = load_seen_policy(
        args.base_checkpoint,
        temporal_seen,
        args.device,
    )

    camera_keys = list(temporal_seen.meta.camera_keys)

    # if len(camera_keys) != 2:
    #     raise RuntimeError(
    #         f"Expected two LIBERO cameras, got {camera_keys}"
    #     )

    state_dim = int(
        temporal_seen.meta.features["observation.state"]["shape"][0]
    )
    action_dim = int(
        temporal_seen.meta.features["action"]["shape"][0]
    )

    language_dim = (
        base_policy.model.vlm_with_expert
        .config.text_config.hidden_size
    )

    adapter = JepaAdapter(
        num_cameras=len(camera_keys),
        language_dim=language_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=args.chunk_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    if args.stage == "adapt":
        if not args.init_adapter:
            raise ValueError("--init_adapter is required for adapt stage")

        checkpoint = torch.load(
            args.init_adapter,
            map_location=device,
        )

        adapter.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

    if args.stage == "pretrain":
        seen_tasks = [
            normalize_task(x)
            for x in plain_seen.meta.tasks.index
        ]

        task_to_episodes = {
            task: matching_episodes(plain_seen, task)
            for task in seen_tasks
        }

        prototype_bank = build_goal_prototypes(
            adapter=adapter,
            plain_dataset=plain_seen,
            task_to_episodes=task_to_episodes,
            device=device,
        )

        frame_indices = all_frame_indices(
            temporal_seen,
            chunk_size=args.chunk_size,
            future_offset=args.future_offset,
        )

        training_dataset = IndexedDataset(
            temporal_seen,
            frame_indices,
        )

        # На pretrain обучаем весь online encoder и action head.
        for parameter in adapter.parameters():
            parameter.requires_grad_(True)

        for parameter in adapter.target_encoder.parameters():
            parameter.requires_grad_(False)

    else:
        temporal_goal = load_temporal_dataset(
            args.data_root,
            "libero_goal",
            chunk_size=args.chunk_size,
            future_offset=args.future_offset,
            video_backend=args.video_backend,
        )

        plain_goal = load_frame_dataset(
            args.data_root,
            "libero_goal",
            video_backend=args.video_backend,
        )

        task_to_episodes = {}

        for task in TARGET_TASKS:
            episodes = matching_episodes(plain_goal, task)

            if len(episodes) < args.budget:
                raise RuntimeError(
                    f"Task {task!r} has only {len(episodes)} episodes"
                )

            # Строго первые N demonstrations.
            task_to_episodes[normalize_task(task)] = episodes[: args.budget]

        prototype_bank = build_goal_prototypes(
            adapter=adapter,
            plain_dataset=plain_goal,
            task_to_episodes=task_to_episodes,
            device=device,
        )

        selected_episodes = [
            episode
            for episodes in task_to_episodes.values()
            for episode in episodes
        ]

        frame_indices = episode_frame_indices(
            temporal_goal,
            selected_episodes,
            chunk_size=args.chunk_size,
            future_offset=args.future_offset,
        )

        training_dataset = IndexedDataset(
            temporal_goal,
            frame_indices,
        )

        # В low-shot режиме сохраняем representation и адаптируем
        # predictor/action head.
        for name, parameter in adapter.named_parameters():
            parameter.requires_grad_(False)

            if name.startswith("predictor"):
                parameter.requires_grad_(True)

            if name.startswith("action_head"):
                parameter.requires_grad_(True)

    loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    loader_iterator = itertools.cycle(loader)

    parameters = [
        parameter
        for parameter in adapter.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    adapter.train()
    base_policy.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        if step % 25 == 0:
            print_memory(f"step={step}")
        raw_batch = next(loader_iterator)

        processed, future_images, tasks = preprocess_window(
            raw_batch,
            preprocessor,
            camera_keys,
        )

        processed = {
            key: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in processed.items()
        }

        current_images = stack_processed_cameras(
            processed,
            camera_keys,
        )

        future_images_tensor = stack_future_cameras(
            future_images,
            camera_keys,
            device,
        )

        with torch.no_grad():
            base_action = predict_base_action(
                base_policy,
                processed,
                step,
            )

            language_embedding = encode_language(
                base_policy,
                processed,
            )

        target_action = processed[ACTION]
        state = processed[OBS_STATE]

        prototypes = goal_tensor(
            tasks,
            prototype_bank,
            device,
        )

        action_mask = processed.get("action_is_pad")
        if action_mask is not None:
            action_mask = ~action_mask.bool()

        output = adapter(
            current_images=current_images,
            future_images=future_images_tensor,
            language_embedding=language_embedding,
            state=state,
            base_action=base_action,
            target_action=target_action,
            goal_prototype=prototypes,
            action_mask=action_mask,
        )



        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if args.stage == "pretrain":
            adapter.update_ema()

        if step % args.log_every == 0:
            print(
                f"step={step:06d} "
                f"loss={output['loss'].item():.5f} "
                f"jepa={output['jepa_loss'].item():.5f} "
                f"action={output['action_loss'].item():.5f}"
            )

        del output
        del current_images
        del future_images_tensor
        del future_images
        del processed
        del raw_batch
        
        if step % 25 == 0:
            gc.collect()
            print_memory(f"after_gc step={step}")

    checkpoint = {
        "state_dict": adapter.state_dict(),
        "prototypes": {
            key: value.cpu()
            for key, value in prototype_bank.items()
        },
        "camera_keys": camera_keys,
        "chunk_size": args.chunk_size,
        "future_offset": args.future_offset,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "language_dim": language_dim,
        "stage": args.stage,
        "budget": args.budget if args.stage == "adapt" else None,
        "seed": args.seed,
    }

    output_path = output_dir / "adapter.pt"
    torch.save(checkpoint, output_path)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": args.stage,
                "budget": args.budget,
                "seed": args.seed,
                "chunk_size": args.chunk_size,
                "future_offset": args.future_offset,
                "camera_keys": camera_keys,
            },
            f,
            indent=2,
        )

    print(f"Saved adapter to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage", choices=["pretrain", "adapt"], required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/libero"))
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--init_adapter", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--video_backend", choices=["pyav", "torchcodec"], default="pyav")
    parser.add_argument("--budget", type=int, choices=[5, 10, 25], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--future_offset", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)

    args = parser.parse_args()

    if args.stage == "pretrain":
        if args.steps is None:
            args.steps = 20_000
        if args.lr is None:
            args.lr = 3e-4
    else:
        if args.budget is None:
            raise ValueError("--budget is required for adapt stage")
        if args.steps is None:
            args.steps = 1_000
        if args.lr is None:
            args.lr = 1e-4

    train(args)


if __name__ == "__main__":
    main()