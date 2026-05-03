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
- [ ] Commit the runner seam.

## Later Chunks

- Add image/video preprocessing helpers around ffmpeg.
- Add a worker-facing runner that calls `TribeModel.from_pretrained`.
- Add optional atlas-backed ROI map construction.
- Add real-model smoke tests gated behind `HF_TOKEN`.
- Add Atherum worker API once the fork contract is stable.
