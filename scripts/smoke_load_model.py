#!/usr/bin/env python3
"""Smoke-test TRIBE v2 model loading.

This script is intentionally small: it verifies that the environment can load
a local TRIBE checkpoint folder without running inference or downloading sample
media. Use --dry-run for a no-import environment check.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
DEFAULT_MODEL_PATH = "/models/tribev2"
DEFAULT_CACHE_FOLDER = "./cache/tribev2"
REQUIRED_MODEL_FILES = ("config.yaml", "best.ckpt")
REPO_ROOT = Path(__file__).resolve().parents[1]


def find_token() -> tuple[str, str] | None:
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a local TRIBE v2 checkpoint folder to verify worker access.",
    )
    parser.add_argument(
        "--model-path",
        "--model-id",
        dest="model_path",
        default=os.environ.get("TRIBE_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Local checkpoint folder containing config.yaml and best.ckpt.",
    )
    parser.add_argument(
        "--cache-folder",
        default=os.environ.get("TRIBE_CACHE_PATH", DEFAULT_CACHE_FOLDER),
        help="Folder used by TRIBE/neuralset feature caches.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device passed to TribeModel.from_pretrained, e.g. "auto", "cpu", "cuda".',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local model files without importing or loading TRIBE.",
    )
    parser.add_argument(
        "--allow-huggingface-download",
        action="store_true",
        help="Allow falling back to a HuggingFace repo id when --model-path is not local.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model_path)
    missing_files = missing_model_files(model_path)
    token = find_token() if args.allow_huggingface_download else None

    if missing_files and not args.allow_huggingface_download:
        print(
            f"Missing local TRIBE model files under {model_path}: "
            f"{', '.join(missing_files)}",
            file=sys.stderr,
        )
        return 2

    if missing_files and args.allow_huggingface_download and not token:
        print(
            "Missing HuggingFace token for remote fallback. Set HF_TOKEN or pass "
            "a local --model-path containing config.yaml and best.ckpt.",
            file=sys.stderr,
        )
        return 2

    cache_folder = Path(args.cache_folder)
    print(f"TRIBE v2 smoke test: model={model_path} cache={cache_folder} device={args.device}")

    if args.dry_run:
        if missing_files:
            print(f"Token preflight: found {token[0]}=<set>")
        else:
            print("Local model preflight: found config.yaml and best.ckpt")
        return 0

    cache_folder.mkdir(parents=True, exist_ok=True)

    if token:
        try:
            from huggingface_hub import login

            login(token=token[1], add_to_git_credential=False)
            print(f"HuggingFace login: using {token[0]}=<set>")
        except TypeError:
            from huggingface_hub import login

            login(token=token[1])
            print(f"HuggingFace login: using {token[0]}=<set>")

    try:
        ensure_repo_root_on_path()
        from tribev2 import TribeModel

        model = TribeModel.from_pretrained(
            model_path,
            cache_folder=cache_folder,
            device=args.device,
        )
    except Exception as exc:  # pragma: no cover - real model path needs gated access.
        print(f"TRIBE v2 model load failed: {exc}", file=sys.stderr)
        return 1

    model_device = getattr(getattr(model, "_model", None), "device", "unknown")
    print(f"TRIBE v2 model loaded successfully on device={model_device}")
    return 0


def missing_model_files(model_path: Path) -> list[str]:
    if not model_path.exists() or not model_path.is_dir():
        return list(REQUIRED_MODEL_FILES)
    return [
        filename
        for filename in REQUIRED_MODEL_FILES
        if not (model_path / filename).is_file()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
