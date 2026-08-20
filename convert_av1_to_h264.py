from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def copy_non_video_files(
    source_root: Path,
    output_root: Path,
) -> None:
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root)
        target = output_root / relative

        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        if path.suffix.lower() == ".mp4":
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def convert_videos(
    source_root: Path,
    output_root: Path,
) -> None:
    videos = sorted(source_root.rglob("*.mp4"))

    print(f"Found {len(videos)} videos")

    for index, source_video in enumerate(videos, start=1):
        relative = source_video.relative_to(source_root)
        output_video = output_root / relative

        output_video.parent.mkdir(parents=True, exist_ok=True)

        if output_video.exists():
            print(f"[{index}/{len(videos)}] skip {relative}")
            continue

        temporary_video = output_video.with_suffix(".tmp.mp4")

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map_metadata",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-an",
            str(temporary_video),
        ]

        print(f"[{index}/{len(videos)}] {relative}")

        subprocess.run(
            command,
            check=True,
        )

        temporary_video.replace(output_video)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("start")
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    print("found")
    args.output.mkdir(parents=True, exist_ok=True)

    for suite in ["libero_90", "libero_goal"]:
        source_suite = args.source / suite
        output_suite = args.output / suite

        if not source_suite.exists():
            print(f"Skip missing suite: {source_suite}")
            continue

        print(f"\nCopying metadata for {suite}")
        copy_non_video_files(source_suite, output_suite)

        print(f"\nConverting videos for {suite}")
        convert_videos(source_suite, output_suite)

    print("\nDone")


if __name__ == "__main__":
    main()