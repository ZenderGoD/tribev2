import unittest


from tribev2.atherum import (
    DEFAULT_TRIBE_MODEL_PATH,
    build_destrieux_roi_vertex_map,
    summarize_vertex_predictions,
)


class AtherumSummaryTests(unittest.TestCase):
    def test_summarizes_temporal_roi_scores_for_atherum_contract(self):
        summary = summarize_vertex_predictions(
            predictions=[
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 4.0, 7.0],
            ],
            roi_vertex_map={
                "V1_V2": [0, 1, 99],
                "DMN": [2, 3],
            },
            modality="video",
        )

        self.assertEqual(summary["modelId"], DEFAULT_TRIBE_MODEL_PATH)
        self.assertEqual(summary["modality"], "video")
        self.assertEqual(summary["predictionShape"], {"timesteps": 2, "vertices": 4})
        self.assertEqual(summary["topRegions"], ["Default Mode Network", "Low-Level Visual Signal"])
        self.assertEqual(summary["metrics"]["predictedSignalStrength"], 0.3214)
        self.assertEqual(summary["metrics"]["visualSalience"], 0.5714)
        self.assertEqual(summary["metrics"]["cognitiveLoadProxy"], 0.5714)

        top_roi = summary["roiData"][0]
        self.assertEqual(top_roi["regionKey"], "DMN")
        self.assertEqual(top_roi["activation"], 0.5714)
        self.assertEqual(top_roi["temporalActivations"], [0.3571, 0.7857])
        self.assertEqual(top_roi["vertexCount"], 2)

        visual_roi = summary["roiData"][1]
        self.assertEqual(visual_roi["regionKey"], "V1_V2")
        self.assertEqual(visual_roi["activation"], 0.0714)
        self.assertEqual(visual_roi["vertexCount"], 2)

    def test_unknown_roi_uses_key_as_label_and_ignores_invalid_vertices(self):
        summary = summarize_vertex_predictions(
            predictions=[[1.0, 2.0, 3.0]],
            roi_vertex_map={"CUSTOM": [-1, 0, 4]},
        )

        self.assertEqual(summary["topRegions"], ["CUSTOM"])
        self.assertEqual(summary["roiData"][0]["label"], "CUSTOM")
        self.assertEqual(summary["roiData"][0]["vertexCount"], 1)

    def test_rejects_empty_predictions(self):
        with self.assertRaises(ValueError):
            summarize_vertex_predictions(predictions=[], roi_vertex_map={"V1_V2": [0]})

    def test_builds_destrieux_roi_vertex_map_from_supplied_atlas(self):
        def fetch_atlas():
            return {
                "map_left": [1, 2, 1, 3],
                "map_right": [2, 1, 3, 1],
                "labels": [b"unknown", b"S_calcarine", b"G_cuneus", b"G_precuneus"],
            }

        roi_map = build_destrieux_roi_vertex_map(
            fetch_atlas=fetch_atlas,
            roi_to_parcels={
                "V1_V2": ["S_calcarine", "G_cuneus"],
                "DMN": ["G_precuneus"],
            },
        )

        self.assertEqual(roi_map["V1_V2"], [0, 1, 2, 4, 5, 7])
        self.assertEqual(roi_map["DMN"], [3, 6])


if __name__ == "__main__":
    unittest.main()
