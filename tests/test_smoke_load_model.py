import importlib.util
import io
import os
import pathlib
import sys
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
    def test_missing_token_fails_before_importing_tribev2(self):
        smoke = load_smoke_module()
        env = {key: value for key, value in os.environ.items() if not key.startswith("HF_")}
        env.pop("HUGGINGFACE_HUB_TOKEN", None)
        env.pop("HUGGINGFACE_TOKEN", None)
        sys.modules.pop("tribev2", None)

        with mock.patch.dict(os.environ, env, clear=True):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(["--dry-run"])

        self.assertEqual(result, 2)
        self.assertNotIn("tribev2", sys.modules)

    def test_skip_token_check_dry_run_succeeds_without_model_download(self):
        smoke = load_smoke_module()

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = smoke.main(["--dry-run", "--skip-token-check"])

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
