# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Modal/FastAPI deployment wrapper for the Atherum TRIBE worker.

The pure ``handle_modal_request`` path is intentionally importable without the
``modal`` package so local contract tests can run before cloud credentials,
model weights, or GPU runtime are available.
"""

from __future__ import annotations

import os
from typing import Any

from tribev2.atherum_worker import FakeTribeModel, run_worker_job


try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]


MODAL_APP_NAME = "atherum-tribe-worker"
MODAL_ENDPOINT_LABEL = "analyze"
MODEL_CACHE_PATH = "/cache/tribev2"
MODEL_VOLUME_NAME = "atherum-tribe-cache"
HF_SECRET_NAME = "atherum-tribe-hf"
TRIBEV2_REPOSITORY_URL = "https://github.com/ZenderGoD/tribev2.git"
TRIBEV2_GIT_REF = "atherum-worker"
DEFAULT_GPU = "L4"
DEFAULT_TIMEOUT_SECONDS = 1800


def modal_available() -> bool:
    return modal is not None


def build_modal_image_requirements() -> dict[str, list[str]]:
    """Return the system and Python dependencies for the Modal runtime image."""

    return {
        "apt": [
            "ffmpeg",
            "git",
        ],
        "pip": [
            "fastapi[standard]",
            "huggingface_hub",
            "nilearn",
            f"git+{TRIBEV2_REPOSITORY_URL}@{TRIBEV2_GIT_REF}",
        ],
    }


def handle_modal_request(
    body: dict[str, Any],
    *,
    fake_model: bool | None = None,
    cache_folder: str = MODEL_CACHE_PATH,
    device: str = "auto",
) -> dict[str, Any]:
    """Run one Atherum worker job and always return the worker response shape."""

    use_fake_model = (
        _env_truthy("ATHERUM_TRIBE_FAKE_MODEL") if fake_model is None else fake_model
    )
    try:
        model = FakeTribeModel() if use_fake_model else None
        return run_worker_job(
            body,
            model=model,
            cache_folder=cache_folder,
            device=device,
        )
    except Exception as exc:
        return _failed_response(body, str(exc))


def _failed_response(body: dict[str, Any], error: str) -> dict[str, Any]:
    source = body if isinstance(body, dict) else {}
    return {
        "jobId": source.get("jobId"),
        "workspaceId": source.get("workspaceId"),
        "projectId": source.get("projectId"),
        "runId": source.get("runId"),
        "status": "failed",
        "error": error,
        "analysis": None,
        "artifacts": [],
    }


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _build_modal_image() -> Any:
    if modal is None:
        return None

    requirements = build_modal_image_requirements()
    image = modal.Image.debian_slim(python_version="3.11").apt_install(
        *requirements["apt"],
    )
    return image.pip_install(*requirements["pip"])


if modal is None:
    app = None
    analyze = None
else:
    app = modal.App(MODAL_APP_NAME)
    _image = _build_modal_image()
    _model_cache_volume = modal.Volume.from_name(
        MODEL_VOLUME_NAME,
        create_if_missing=True,
    )
    _hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

    @app.function(
        image=_image,
        volumes={MODEL_CACHE_PATH: _model_cache_volume},
        timeout=DEFAULT_TIMEOUT_SECONDS,
        gpu=DEFAULT_GPU,
        secrets=[_hf_secret],
    )
    @modal.fastapi_endpoint(
        method="POST",
        label=MODAL_ENDPOINT_LABEL,
        requires_proxy_auth=True,
    )
    def analyze(body: dict[str, Any]) -> dict[str, Any]:
        return handle_modal_request(body, cache_folder=MODEL_CACHE_PATH)
