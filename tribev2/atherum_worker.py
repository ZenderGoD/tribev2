# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Local Atherum TRIBE worker entrypoint.

This module defines the JSON job contract Atherum can call later from Convex or
an API gateway. It is intentionally runnable with ``--fake-model`` so the worker
surface can be tested before gated model weights are available.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import sys
from typing import Any

from tribev2.atherum import AtherumTribeRunner, build_destrieux_roi_vertex_map


SUPPORTED_MODALITIES = {"text", "audio", "video", "multimodal"}


class FakeTribeModel:
    """Deterministic tiny model for worker contract tests."""

    def get_events_dataframe(self, **kwargs: str) -> dict[str, dict[str, str]]:
        return {"events": kwargs}

    def predict(self, *, events: Any) -> tuple[list[list[float]], list[str]]:
        return [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 5.0],
        ], ["fake-segment-1", "fake-segment-2"]


def validate_worker_job(job: dict[str, Any]) -> dict[str, Any]:
    required_string_fields = ("jobId", "workspaceId", "projectId", "runId", "modality")
    normalized: dict[str, Any] = {}

    for field in required_string_fields:
        value = job.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")
        normalized[field] = value.strip()

    if normalized["modality"] not in SUPPORTED_MODALITIES:
        raise ValueError(f"unsupported modality: {normalized['modality']}")

    input_spec = job.get("input")
    if not isinstance(input_spec, dict):
        raise ValueError("input must be an object")
    path = input_spec.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("input.path must be a nonempty string")
    normalized["input"] = {"path": path.strip()}

    roi_vertex_map = job.get("roiVertexMap")
    if roi_vertex_map is not None:
        if not isinstance(roi_vertex_map, dict):
            raise ValueError("roiVertexMap must be an object when provided")
        normalized["roiVertexMap"] = _coerce_roi_vertex_map(roi_vertex_map)

    model_id = job.get("modelId", "facebook/tribev2")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("modelId must be a nonempty string")
    normalized["modelId"] = model_id.strip()

    max_regions = job.get("maxRegions", 5)
    if not isinstance(max_regions, int) or max_regions < 1:
        raise ValueError("maxRegions must be a positive integer")
    normalized["maxRegions"] = max_regions

    return normalized


def run_worker_job(
    job: dict[str, Any],
    *,
    model: Any | None = None,
    cache_folder: str = "./cache/tribev2",
    device: str = "auto",
) -> dict[str, Any]:
    normalized = validate_worker_job(job)
    roi_vertex_map = normalized.get("roiVertexMap") or build_destrieux_roi_vertex_map()
    runner = AtherumTribeRunner(
        model=model,
        model_id=normalized["modelId"],
        cache_folder=cache_folder,
        device=device,
    )

    analysis = runner.analyze_path(
        path=normalized["input"]["path"],
        modality=normalized["modality"],
        roi_vertex_map=roi_vertex_map,
        max_regions=normalized["maxRegions"],
    )

    return {
        "jobId": normalized["jobId"],
        "workspaceId": normalized["workspaceId"],
        "projectId": normalized["projectId"],
        "runId": normalized["runId"],
        "status": "completed",
        "error": None,
        "analysis": analysis,
        "artifacts": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Atherum TRIBE worker job.")
    parser.add_argument("job", help="Path to the Atherum TRIBE job JSON file.")
    parser.add_argument(
        "--output",
        help="Optional path to write the response JSON. Defaults to stdout.",
    )
    parser.add_argument(
        "--fake-model",
        action="store_true",
        help="Use a deterministic no-download fake model for contract testing.",
    )
    parser.add_argument(
        "--cache-folder",
        default="./cache/tribev2",
        help="Cache folder passed to TribeModel.from_pretrained in real mode.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device passed to TribeModel.from_pretrained, e.g. "auto", "cpu", "cuda".',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        model = FakeTribeModel() if args.fake_model else None
        response = run_worker_job(
            job,
            model=model,
            cache_folder=args.cache_folder,
            device=args.device,
        )
    except Exception as exc:
        print(f"Atherum TRIBE worker failed: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(response, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


def _coerce_roi_vertex_map(raw: dict[str, Any]) -> dict[str, list[int]]:
    roi_vertex_map: dict[str, list[int]] = {}
    for key, vertices in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError("roiVertexMap keys must be nonempty strings")
        if not isinstance(vertices, Iterable) or isinstance(vertices, (str, bytes)):
            raise ValueError(f"roiVertexMap.{key} must be an array of vertex indices")
        roi_vertex_map[key] = [int(vertex) for vertex in vertices]
    return roi_vertex_map


if __name__ == "__main__":
    raise SystemExit(main())
