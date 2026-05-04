import unittest

from tribev2.atherum import AtherumTribeRunner, DEFAULT_TRIBE_MODEL_PATH


class FakeModel:
    def __init__(self):
        self.events_kwargs = None
        self.predict_events = None

    def get_events_dataframe(self, **kwargs):
        self.events_kwargs = kwargs
        return {"events": kwargs}

    def predict(self, *, events):
        self.predict_events = events
        return [[0.0, 1.0], [1.0, 3.0]], ["segment-1", "segment-2"]


class FakeTribeModelLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return FakeModel()


class AtherumTribeRunnerTests(unittest.TestCase):
    def test_defaults_to_local_model_artifact_path(self):
        runner = AtherumTribeRunner(model=FakeModel())

        self.assertEqual(runner.model_id, DEFAULT_TRIBE_MODEL_PATH)

    def test_load_model_passes_config_update_to_tribe_model(self):
        import tribev2

        config_update = {
            "data.num_workers": 0,
            "data.video_feature.image.device": "cpu",
        }
        runner = AtherumTribeRunner(
            model_id="/models/tribev2",
            cache_folder="/cache/tribev2",
            device="cpu",
            config_update=config_update,
        )
        FakeTribeModelLoader.calls = []

        original = tribev2.__dict__.get("TribeModel")
        had_original = "TribeModel" in tribev2.__dict__
        tribev2.__dict__["TribeModel"] = FakeTribeModelLoader
        try:
            runner.load_model()
        finally:
            if had_original:
                tribev2.__dict__["TribeModel"] = original
            else:
                del tribev2.__dict__["TribeModel"]

        args, kwargs = FakeTribeModelLoader.calls[0]
        self.assertEqual(args, ("/models/tribev2",))
        self.assertEqual(kwargs["cache_folder"], "/cache/tribev2")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["config_update"], config_update)

    def test_analyze_path_maps_video_input_to_tribe_and_returns_summary(self):
        model = FakeModel()
        runner = AtherumTribeRunner(model=model)

        summary = runner.analyze_path(
            path="/tmp/stimulus.mp4",
            modality="video",
            roi_vertex_map={"V1_V2": [0], "DMN": [1]},
        )

        self.assertEqual(model.events_kwargs, {"video_path": "/tmp/stimulus.mp4"})
        self.assertEqual(model.predict_events, {"events": {"video_path": "/tmp/stimulus.mp4"}})
        self.assertEqual(summary["modality"], "video")
        self.assertEqual(summary["predictionShape"], {"timesteps": 2, "vertices": 2})
        self.assertEqual(summary["topRegions"][0], "Default Mode Network")

    def test_analyze_path_maps_text_input_to_tribe(self):
        model = FakeModel()
        runner = AtherumTribeRunner(model=model)

        runner.analyze_path(
            path="/tmp/hook.txt",
            modality="text",
            roi_vertex_map={"DMN": [1]},
        )

        self.assertEqual(model.events_kwargs, {"text_path": "/tmp/hook.txt"})

    def test_rejects_image_until_preprocessed_to_video(self):
        runner = AtherumTribeRunner(model=FakeModel())

        with self.assertRaises(ValueError):
            runner.analyze_path(
                path="/tmp/image.png",
                modality="image",
                roi_vertex_map={"V1_V2": [0]},
            )


if __name__ == "__main__":
    unittest.main()
