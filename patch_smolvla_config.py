from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    if not (args.base / "config.json").exists():
        raise FileNotFoundError(args.base / "config.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        args.base,
        args.output,
        dirs_exist_ok=True,
    )

    dataset = LeRobotDataset(
        repo_id="nvidia/LIBERO_LeRobot_v3",
        root=args.dataset_root,
        video_backend="pyav",
        return_uint8=False,
    )

    camera_keys = list(dataset.meta.camera_keys)

    input_features = {}

    for key in camera_keys:
        feature = dataset.meta.features[key]
        input_features[key] = {
            "type": "VISUAL",
            "shape": list(feature["shape"]),
        }

    state_feature = dataset.meta.features["observation.state"]

    input_features["observation.state"] = {
        "type": "STATE",
        "shape": list(state_feature["shape"]),
    }

    action_feature = dataset.meta.features["action"]

    output_features = {
        "action": {
            "type": "ACTION",
            "shape": list(action_feature["shape"]),
        }
    }

    config_path = args.output / "config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["input_features"] = input_features
    config["output_features"] = output_features

    # Убираем исходный Hub ID, который ломается на Windows.
    config["pretrained_path"] = None

    # SmolVLA использует фиксированные максимальные размеры внутренних
    # проекций. Они должны оставаться не меньше LIBERO-размеров.
    config["max_state_dim"] = max(
        int(config.get("max_state_dim", 32)),
        int(state_feature["shape"][0]),
    )

    config["max_action_dim"] = max(
        int(config.get("max_action_dim", 32)),
        int(action_feature["shape"][0]),
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Saved:", config_path)
    print("camera keys:", camera_keys)
    print("state:", state_feature["shape"])
    print("action:", action_feature["shape"])


if __name__ == "__main__":
    main()