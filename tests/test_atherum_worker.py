import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tribev2.atherum import DEFAULT_TRIBE_MODEL_PATH
from tribev2.atherum_worker import FakeTribeModel, run_worker_job, validate_worker_job


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


class AtherumWorkerTests(unittest.TestCase):
    def test_run_worker_job_with_fake_model_returns_atherum_contract(self):
        response = run_worker_job(example_job(), model=FakeTribeModel())

        self.assertEqual(response["jobId"], "tribe-job-1")
        self.assertEqual(response["workspaceId"], "workspace-1")
        self.assertEqual(response["projectId"], "project-1")
        self.assertEqual(response["runId"], "prediction-run-1")
        self.assertEqual(response["status"], "completed")
        self.assertIsNone(response["error"])
        self.assertEqual(response["artifacts"], [])
        self.assertEqual(response["analysis"]["modelId"], DEFAULT_TRIBE_MODEL_PATH)
        self.assertEqual(response["analysis"]["modality"], "video")
        self.assertEqual(response["analysis"]["topRegions"][0], "Default Mode Network")

    def test_validate_worker_job_rejects_missing_input_path(self):
        job = example_job()
        job["input"] = {}

        with self.assertRaises(ValueError):
            validate_worker_job(job)

    def test_validate_worker_job_uses_local_model_path_when_not_supplied(self):
        normalized = validate_worker_job(example_job())

        self.assertEqual(normalized["modelId"], DEFAULT_TRIBE_MODEL_PATH)

    def test_validate_worker_job_accepts_explicit_local_model_path(self):
        job = example_job()
        job["modelId"] = "/private/models/tribev2"

        normalized = validate_worker_job(job)

        self.assertEqual(normalized["modelId"], "/private/models/tribev2")

    def test_validate_worker_job_accepts_json_config_update(self):
        job = example_job()
        job["configUpdate"] = {
            "data.num_workers": 0,
            "data.video_feature.image.device": "cpu",
            "data.video_feature.num_frames": 8,
        }

        normalized = validate_worker_job(job)

        self.assertEqual(normalized["configUpdate"], job["configUpdate"])

    def test_validate_worker_job_rejects_non_object_config_update(self):
        job = example_job()
        job["configUpdate"] = ["data.num_workers=0"]

        with self.assertRaises(ValueError):
            validate_worker_job(job)

    def test_cli_writes_fake_model_output_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_path = Path(tmp) / "job.json"
            output_path = Path(tmp) / "result.json"
            job_path.write_text(json.dumps(example_job()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tribev2.atherum_worker",
                    str(job_path),
                    "--fake-model",
                    "--output",
                    str(output_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["analysis"]["predictionShape"], {"timesteps": 2, "vertices": 4})


if __name__ == "__main__":
    unittest.main()
