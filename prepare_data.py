from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "nvidia/LIBERO_LeRobot_v3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/libero"))
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=[
            "libero_90/**",
            "libero_goal/**",
        ],
    )

    manifest = {
        "repo_id": REPO_ID,
        "revision": args.revision,
        "local_path": str(Path(snapshot_path).resolve()),
        "suites": {
            "seen": str((args.output / "libero_90").resolve()),
            "goal": str((args.output / "libero_goal").resolve()),
        },
    }

    with open(args.output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()