import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "smoke_load_model.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_load_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SmokeLoadModelTests(unittest.TestCase):
    def test_ensure_repo_root_on_path_adds_package_parent(self):
        smoke = load_smoke_module()
        repo_root = str(SCRIPT_PATH.parents[1])

        with mock.patch.object(sys, "path", [str(SCRIPT_PATH.parent)]):
            smoke.ensure_repo_root_on_path()

            self.assertEqual(sys.path[0], repo_root)

    def test_missing_local_model_files_fail_before_importing_tribev2(self):
        smoke = load_smoke_module()
        env = {key: value for key, value in os.environ.items() if not key.startswith("HF_")}
        env.pop("HUGGINGFACE_HUB_TOKEN", None)
        env.pop("HUGGINGFACE_TOKEN", None)
        sys.modules.pop("tribev2", None)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, env, clear=True):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = smoke.main(["--dry-run", "--model-path", tmp])

        self.assertEqual(result, 2)
        self.assertNotIn("tribev2", sys.modules)

    def test_dry_run_succeeds_with_local_model_files_and_no_hf_token(self):
        smoke = load_smoke_module()

        with tempfile.TemporaryDirectory() as tmp:
            model_path = pathlib.Path(tmp)
            (model_path / "config.yaml").write_text("config: test\n", encoding="utf-8")
            (model_path / "best.ckpt").write_bytes(b"placeholder")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(["--dry-run", "--model-path", str(model_path)])

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
