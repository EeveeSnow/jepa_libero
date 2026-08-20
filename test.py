import torch

from adapter_runner import dataset_action_to_libero

print(torch.cuda.is_available())
dataset_action = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # open
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # closed
    ]
)

print(dataset_action_to_libero(dataset_action))


# lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 --dataset.root=./data/libero/libero_90 --dataset.video_backend=pyav --policy.device=cuda --policy.push_to_hub=false --output_dir=outputs/smolvla_seen --job_name=smolvla_seen --batch_size=32 --steps=100000

