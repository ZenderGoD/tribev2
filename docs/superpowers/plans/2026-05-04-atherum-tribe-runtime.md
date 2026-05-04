# Atherum TRIBE Runtime Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable Atherum-facing TRIBE runtime layer that can be tested before downloading model weights.

**Architecture:** Keep upstream TRIBE model code intact where possible. Add lightweight package imports, a pure-Python Atherum summary layer, then later add worker-facing inference helpers that call `TribeModel` only when real weights are available.

**Tech Stack:** Python 3.11+, standard-library unit tests for no-download paths, TRIBE v2/HuggingFace for later real inference.

---

## Chunk 1: Lightweight Import And Summary Layer

### Task 1: Make package import lightweight

**Files:**
- Modify: `tribev2/__init__.py`
- Test: `tests/test_package_import.py`

- [x] Write a failing test proving `import tribev2` does not import heavy runtime dependencies.
- [x] Run the test and verify it fails against the current eager `TribeModel` import.
- [x] Replace the eager import with a lazy `__getattr__` that preserves `from tribev2 import TribeModel`.
- [x] Run the targeted test and existing smoke tests.
- [x] Commit the import fix.

### Task 2: Add Atherum ROI summary helpers

**Files:**
- Create: `tribev2/atherum.py`
- Test: `tests/test_atherum_summary.py`

- [x] Write failing tests for temporal ROI extraction, invalid-vertex skipping, and stable Atherum output keys.
- [x] Run tests and verify they fail because the module is missing.
- [x] Implement pure-Python summary helpers with no model download and no eager heavy imports.
- [x] Run targeted tests and smoke tests.
- [x] Commit the summary layer.

### Task 3: Add worker-facing runner seam

**Files:**
- Modify: `tribev2/atherum.py`
- Test: `tests/test_atherum_runner.py`

- [x] Write failing tests for modality-to-TRIBE path mapping and fake-model summary execution.
- [x] Run tests and verify they fail because `AtherumTribeRunner` is missing.
- [x] Implement a lazy-loading runner that accepts an injected model in tests.
- [x] Run targeted tests and smoke tests.
- [x] Commit the runner seam.

### Task 4: Add media preprocessing command helpers

**Files:**
- Create: `tribev2/atherum_preprocess.py`
- Test: `tests/test_atherum_preprocess.py`

- [x] Write failing tests for static-image video conversion and worker video preparation commands.
- [x] Run tests and verify they fail because `atherum_preprocess` is missing.
- [x] Implement ffmpeg command builders and thin subprocess wrappers.
- [x] Run targeted tests and smoke tests.
- [x] Commit the preprocessing helpers.

### Task 5: Add atlas-backed ROI map builder

**Files:**
- Modify: `tribev2/atherum.py`
- Test: `tests/test_atherum_summary.py`

- [x] Write a failing test using a fake Destrieux atlas fixture.
- [x] Run tests and verify they fail because the builder is missing.
- [x] Implement lazy atlas loading with sorted, stable ROI vertex indices.
- [x] Run targeted tests and smoke tests.
- [x] Commit the ROI map builder.

## Chunk 2: Local Worker Contract

### Task 6: Add no-download Atherum worker CLI

**Files:**
- Create: `tribev2/atherum_worker.py`
- Test: `tests/test_atherum_worker.py`

- [x] Write failing tests for job validation, fake-model worker output, and CLI JSON writing.
- [x] Run tests and verify they fail because `atherum_worker` is missing.
- [x] Implement the worker job contract, fake model, `run_worker_job`, and `python -m tribev2.atherum_worker`.
- [x] Run targeted tests and the no-model suite.
- [x] Commit the worker CLI.

## Chunk 3: Cloud Worker Surface

### Task 7: Add Modal/FastAPI deployment wrapper

**Files:**
- Create: `tribev2/atherum_modal.py`
- Test: `tests/test_atherum_modal.py`

- [x] Write failing tests for optional Modal import, fake-model request handling, failed response contracts, and runtime dependency metadata.
- [x] Run tests and verify they fail because `atherum_modal` is missing.
- [x] Implement the Modal/FastAPI wrapper around the local worker contract without downloading model weights.
- [x] Run targeted tests and the no-model suite.
- [x] Commit the Modal wrapper.

## Chunk 4: Local Model Artifact Runtime

### Task 8: Remove Hugging Face runtime assumptions

**Files:**
- Modify: `tribev2/atherum.py`
- Modify: `tribev2/atherum_worker.py`
- Modify: `tribev2/atherum_modal.py`
- Modify: `scripts/smoke_load_model.py`
- Modify: `.gitignore`
- Tests: `tests/test_atherum_summary.py`, `tests/test_atherum_runner.py`, `tests/test_atherum_worker.py`, `tests/test_atherum_modal.py`, `tests/test_smoke_load_model.py`

- [x] Move downloaded model files out of git to `/Users/bishalbanerjee/zeus/models/tribev2`.
- [x] Write failing tests for local model defaults, local smoke preflight, and no HF secret in the worker wrapper.
- [x] Change Atherum worker defaults from `facebook/tribev2` to local `/models/tribev2`, with `TRIBE_MODEL_PATH` override support.
- [x] Update Modal wiring to mount a private model volume and cache volume without an HF secret.
- [x] Update the smoke script to validate local `config.yaml` and `best.ckpt` before importing TRIBE.
- [x] Run full no-model tests and local model preflight.
- [x] Commit the local artifact runtime changes.

## Chunk 5: Extractor Cache And Real Worker Video

### Task 9: Add worker runtime config overrides

**Files:**
- Modify: `tribev2/atherum.py`
- Modify: `tribev2/atherum_worker.py`
- Tests: `tests/test_atherum_runner.py`, `tests/test_atherum_worker.py`

- [x] Write failing tests for passing `configUpdate` through the worker to `TribeModel.from_pretrained`.
- [x] Implement `configUpdate` validation and runner forwarding.
- [x] Run targeted tests.
- [x] Run the real video worker CLI with offline V-JEPA2 cache and CPU-safe overrides.
- [ ] Run full test suite and commit the worker config override changes.

### Task 10: Cache extractor weights for local/offline runtime

**Local artifacts:**
- `/Users/bishalbanerjee/zeus/models/hf-cache`
- `/Users/bishalbanerjee/zeus/models/vjepa2-vitg-fpc64-256`

- [x] Download and validate `facebook/vjepa2-vitg-fpc64-256` for offline video inference.
- [x] Download and validate `facebook/w2v-bert-2.0` for offline audio inference.
- [ ] Download `meta-llama/Llama-3.2-3B` after account access is approved.

## Later Chunks

- Deploy the worker with mounted model/cache volumes.
- Add Atherum-side Convex/API client that calls the protected Modal endpoint.
