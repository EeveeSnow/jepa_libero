from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from jepa_adapter import JepaAdapter, encode_language
from train_adapter import stack_processed_cameras


def dataset_action_to_libero(action: torch.Tensor) -> torch.Tensor:
    """
    В nvidia/LIBERO_LeRobot_v3:

        0 = closed
        1 = open

    В LIBERO environment:

        -1 = open
        +1 = closed

    Поэтому:

        env_gripper = 1 - 2 * dataset_gripper
    """
    action = action.clone()

    gripper = action[..., -1]
    gripper = gripper.clamp(0.0, 1.0)
    action[..., -1] = 1.0 - 2.0 * gripper

    return action


class AdapterRunner:
    def __init__(
        self,
        *,
        base_policy,
        preprocessor,
        postprocessor,
        adapter: JepaAdapter,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.base_policy = base_policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.adapter = adapter.to(device)
        self.device = torch.device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.adapter.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        self.prototypes = {
            key: value.to(self.device)
            for key, value in checkpoint["prototypes"].items()
        }

        self.camera_keys = list(checkpoint["camera_keys"])
        self.adapter.eval()
        self.base_policy.eval()

    @torch.no_grad()
    def predict(
        self,
        raw_observation: dict[str, Any],
        task: str,
    ) -> torch.Tensor:
        """
        raw_observation должен содержать:

            observation.images.image
            observation.images.wrist_image
            observation.state

        Изображения допускаются в HWC или CHW формате.
        """
        batch = {}

        for key, value in raw_observation.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    value = value.unsqueeze(0)
                elif value.ndim == 1:
                    value = value.unsqueeze(0)

                batch[key] = value

        batch["task"] = [task]

        processed = self.preprocessor(batch)

        processed = {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in processed.items()
        }

        language_embedding = encode_language(
            self.base_policy,
            processed,
        )

        self.base_policy.reset()

        base_action = self.base_policy.predict_action_chunk(
            processed,
        )

        current_images = stack_processed_cameras(
            processed,
            self.camera_keys,
        )

        if task not in self.prototypes:
            raise KeyError(
                f"No prototype for task {task!r}. "
                "Use an adapter checkpoint trained for this target task."
            )

        goal_prototype = self.prototypes[task].unsqueeze(0)

        predicted_action = self.adapter.act(
            current_images=current_images,
            language_embedding=language_embedding,
            state=processed["observation.state"],
            base_action=base_action,
            goal_prototype=goal_prototype,
        )

        predicted_action = self.postprocessor(predicted_action)
        predicted_action = dataset_action_to_libero(predicted_action)

        return predicted_action