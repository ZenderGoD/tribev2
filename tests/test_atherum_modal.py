import builtins
import importlib
import sys
import unittest


def example_job():
    return {
        "jobId": "tribe-job-1",
        "workspaceId": "workspace-1",
        "projectId": "project-1",
        "runId": "prediction-run-1",
        "modality": "video",
        "input": {"path": "/tmp/stimulus.mp4"},
        "roiVertexMap": {
            "V1_V2": [0, 1],
            "DMN": [2, 3],
        },
        "maxRegions": 2,
    }


def import_without_modal():
    sys.modules.pop("tribev2.atherum_modal", None)
    sys.modules.pop("modal", None)
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "modal" or name.startswith("modal."):
            raise ImportError("modal intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        return importlib.import_module("tribev2.atherum_modal")
    finally:
        builtins.__import__ = original_import


class AtherumModalTests(unittest.TestCase):
    def test_module_import_does_not_require_modal_package(self):
        module = import_without_modal()

        self.assertFalse(module.modal_available())
        self.assertIsNone(module.app)
        self.assertIsNone(module.analyze)
        self.assertEqual(module.MODAL_APP_NAME, "atherum-tribe-worker")
        self.assertEqual(module.MODAL_ENDPOINT_LABEL, "analyze")
        self.assertEqual(module.MODEL_PATH, "/models/tribev2")
        self.assertEqual(module.MODEL_CACHE_PATH, "/cache/tribev2")
        self.assertEqual(module.MODEL_VOLUME_NAME, "atherum-tribe-model")
        self.assertEqual(module.CACHE_VOLUME_NAME, "atherum-tribe-cache")
        self.assertFalse(hasattr(module, "HF_SECRET_NAME"))

    def test_handle_modal_request_runs_worker_contract_with_fake_model(self):
        module = import_without_modal()

        response = module.handle_modal_request(example_job(), fake_model=True)

        self.assertEqual(response["jobId"], "tribe-job-1")
        self.assertEqual(response["status"], "completed")
        self.assertIsNone(response["error"])
        self.assertEqual(response["analysis"]["modelId"], "/models/tribev2")
        self.assertEqual(
            response["analysis"]["predictionShape"],
            {"timesteps": 2, "vertices": 4},
        )

    def test_handle_modal_request_returns_failed_contract_for_invalid_job(self):
        module = import_without_modal()

        job = example_job()
        job["input"] = {}

        response = module.handle_modal_request(job, fake_model=True)

        self.assertEqual(response["jobId"], "tribe-job-1")
        self.assertEqual(response["workspaceId"], "workspace-1")
        self.assertEqual(response["projectId"], "project-1")
        self.assertEqual(response["runId"], "prediction-run-1")
        self.assertEqual(response["status"], "failed")
        self.assertIn("input.path", response["error"])
        self.assertIsNone(response["analysis"])
        self.assertEqual(response["artifacts"], [])

    def test_modal_image_requirements_document_runtime_surface(self):
        module = import_without_modal()

        requirements = module.build_modal_image_requirements()

        self.assertIn("fastapi[standard]", requirements["pip"])
        self.assertNotIn("huggingface_hub", requirements["pip"])
        self.assertIn("ffmpeg", requirements["apt"])
        self.assertIn("git", requirements["apt"])


if __name__ == "__main__":
    unittest.main()
