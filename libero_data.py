from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ID = "nvidia/LIBERO_LeRobot_v3"

TARGET_TASKS = [
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
]


def normalize_task(task: str) -> str:
    return " ".join(str(task).strip().split())


def load_temporal_dataset(
    root: str | Path,
    suite: str,
    *,
    chunk_size: int = 16,
    future_offset: int = 8,
    video_backend: str = "pyav",
) -> LeRobotDataset:
    root = Path(root) / suite

    # Сначала читаем только metadata, чтобы узнать camera keys и fps.
    metadata_dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=root,
        video_backend=video_backend,
        return_uint8=False,
    )

    fps = float(metadata_dataset.meta.fps)
    camera_keys = list(metadata_dataset.meta.camera_keys)

    delta_timestamps: dict[str, list[float]] = {
        "action": [
            i / fps
            for i in range(chunk_size)
        ],
    }

    for camera_key in camera_keys:
        delta_timestamps[camera_key] = [
            0.0,
            future_offset / fps,
        ]

    return LeRobotDataset(
        repo_id=REPO_ID,
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        return_uint8=False,
    )

def load_frame_dataset(
    root: str | Path,
    suite: str,
    *,
    video_backend: str = "pyav",
) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id=REPO_ID,
        root=Path(root) / suite,
        video_backend=video_backend,
        return_uint8=False,
    )


def _row(dataset: LeRobotDataset, episode_index: int) -> dict[str, Any]:
    episodes = dataset.meta.episodes

    try:
        return dict(episodes[episode_index])
    except Exception:
        return dict(episodes.iloc[episode_index])


def _row_task_matches(
    dataset: LeRobotDataset,
    episode_index: int,
    task: str,
) -> bool:
    wanted = normalize_task(task)
    wanted_index = dataset.meta.get_task_index(task)
    row = _row(dataset, episode_index)

    # Возможный формат v3: tasks = ["instruction"]
    tasks = row.get("tasks")
    if isinstance(tasks, str):
        if normalize_task(tasks) == wanted:
            return True

    if isinstance(tasks, Iterable) and not isinstance(tasks, (str, bytes, dict)):
        for value in tasks:
            if isinstance(value, str) and normalize_task(value) == wanted:
                return True
            if wanted_index is not None and value == wanted_index:
                return True

    # В некоторых версиях metadata task_index лежит непосредственно в episode row.
    if wanted_index is not None:
        for key in ("task_index", "task_id"):
            if row.get(key) == wanted_index:
                return True

    # Надёжный fallback: проверяем первую строку самого эпизода.
    start = int(row["dataset_from_index"])
    raw = dataset.get_raw_item(start)
    raw_task_index = raw.get("task_index")

    if wanted_index is not None:
        try:
            return int(raw_task_index) == int(wanted_index)
        except Exception:
            pass

    raw_task = raw.get("task")
    if isinstance(raw_task, str):
        return normalize_task(raw_task) == wanted

    return False


def matching_episodes(
    dataset: LeRobotDataset,
    task: str,
) -> list[int]:
    result = []

    for episode_index in range(dataset.meta.total_episodes):
        if _row_task_matches(dataset, episode_index, task):
            result.append(episode_index)

    return sorted(result)


def episode_frame_indices(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    *,
    chunk_size: int,
    future_offset: int,
) -> list[int]:
    result: list[int] = []

    # Нельзя брать последние кадры эпизода: для них action chunk
    # или future target будут padded.
    required_future = max(chunk_size, future_offset + 1)

    for episode_index in episode_indices:
        row = _row(dataset, episode_index)
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])

        last_valid_start = end - required_future

        if last_valid_start < start:
            continue

        result.extend(range(start, last_valid_start + 1))

    return result


def all_frame_indices(
    dataset: LeRobotDataset,
    *,
    chunk_size: int,
    future_offset: int,
) -> list[int]:
    return episode_frame_indices(
        dataset,
        list(range(dataset.meta.total_episodes)),
        chunk_size=chunk_size,
        future_offset=future_offset,
    )


class IndexedDataset(Dataset):
    def __init__(
        self,
        dataset: LeRobotDataset,
        indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]