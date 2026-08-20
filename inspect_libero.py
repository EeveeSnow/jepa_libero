from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DATASET_ROOT = Path("C:/ml/jepa_libero/data/libero/libero_90")


dataset = LeRobotDataset(
    repo_id="nvidia/LIBERO_LeRobot_v3",
    root=DATASET_ROOT,
    video_backend="pyav",
    return_uint8=False,
)

print("episodes:", dataset.meta.total_episodes)
print("frames:", dataset.meta.total_frames)
print("fps:", dataset.meta.fps)
print("camera keys:", list(dataset.meta.camera_keys))
print("tasks:")
for task in dataset.meta.tasks.index:
    print("  ", repr(task))

print("\nfeatures:")
for key, feature in dataset.meta.features.items():
    print(key, feature)

sample = dataset[0]

print("\nsample keys:")
for key, value in sample.items():
    if hasattr(value, "shape"):
        print(key, tuple(value.shape), value.dtype)
    else:
        print(key, type(value))

if "action" in sample:
    action = sample["action"]
    print("\naction shape:", tuple(action.shape))
    print("action min:", action.min().item())
    print("action max:", action.max().item())
    print("gripper values:", action[..., -1].unique())