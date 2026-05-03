#!/usr/bin/env python3
"""Smoke-test TRIBE v2 model loading.

This script is intentionally small: it verifies that the environment can load
the gated HuggingFace checkpoint without running inference or downloading sample
media. Use --dry-run for a no-download environment check.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def find_token() -> tuple[str, str] | None:
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load facebook/tribev2 from HuggingFace to verify local access.",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/tribev2",
        help="HuggingFace model id or local checkpoint folder.",
    )
    parser.add_argument(
        "--cache-folder",
        default="./cache/tribev2",
        help="Folder used by TRIBE/neuralset/HuggingFace caches.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device passed to TribeModel.from_pretrained, e.g. "auto", "cpu", "cuda".',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and token presence without importing or loading TRIBE.",
    )
    parser.add_argument(
        "--skip-token-check",
        action="store_true",
        help="Skip the HuggingFace token preflight, useful if auth is already cached.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = find_token()

    if not token and not args.skip_token_check:
        print(
            "Missing HuggingFace token. Set HF_TOKEN after access is approved for "
            "facebook/tribev2 and meta-llama/Llama-3.2-3B, or pass "
            "--skip-token-check if credentials are already cached.",
            file=sys.stderr,
        )
        return 2

    cache_folder = Path(args.cache_folder)
    print(f"TRIBE v2 smoke test: model={args.model_id} cache={cache_folder} device={args.device}")

    if args.dry_run:
        if token:
            print(f"Token preflight: found {token[0]}=<set>")
        else:
            print("Token preflight: skipped")
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
        from tribev2 import TribeModel

        model = TribeModel.from_pretrained(
            args.model_id,
            cache_folder=cache_folder,
            device=args.device,
        )
    except Exception as exc:  # pragma: no cover - real model path needs gated access.
        print(f"TRIBE v2 model load failed: {exc}", file=sys.stderr)
        return 1

    model_device = getattr(getattr(model, "_model", None), "device", "unknown")
    print(f"TRIBE v2 model loaded successfully on device={model_device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
