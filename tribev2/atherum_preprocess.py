# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Atherum-oriented media preprocessing helpers for TRIBE inference."""

from __future__ import annotations

import subprocess


def build_image_to_video_command(
    *,
    input_path: str,
    output_path: str,
    duration_sec: int = 4,
    width: int = 640,
    height: int = 360,
) -> list[str]:
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    return [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-t",
        str(duration_sec),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        vf,
        output_path,
    ]


def build_prepare_video_command(
    *,
    input_path: str,
    output_path: str,
    max_duration_sec: int = 60,
    fps: int = 4,
    strip_audio: bool = True,
) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        str(max_duration_sec),
    ]
    if strip_audio:
        command.append("-an")
    command.extend(
        [
            "-vf",
            f"fps={fps}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            output_path,
        ]
    )
    return command


def image_to_video(
    input_path: str,
    output_path: str,
    *,
    duration_sec: int = 4,
    width: int = 640,
    height: int = 360,
) -> None:
    command = build_image_to_video_command(
        input_path=input_path,
        output_path=output_path,
        duration_sec=duration_sec,
        width=width,
        height=height,
    )
    subprocess.run(command, check=True, capture_output=True)


def prepare_video(
    input_path: str,
    output_path: str,
    *,
    max_duration_sec: int = 60,
    fps: int = 4,
    strip_audio: bool = True,
) -> None:
    command = build_prepare_video_command(
        input_path=input_path,
        output_path=output_path,
        max_duration_sec=max_duration_sec,
        fps=fps,
        strip_audio=strip_audio,
    )
    subprocess.run(command, check=True, capture_output=True)
