# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Atherum-facing helpers for bounded TRIBE v2 result summaries.

The functions in this module intentionally avoid importing torch, numpy, or the
model runtime. They can run in API/control-plane tests and can also be reused by
GPU workers after real TRIBE predictions are available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
from statistics import mean
from typing import Any


DEFAULT_TRIBE_MODEL_PATH = "/models/tribev2"
DEFAULT_TRIBE_CACHE_FOLDER = "./cache/tribev2"
TRIBE_MODEL_PATH_ENV = "TRIBE_MODEL_PATH"
TRIBE_CACHE_PATH_ENV = "TRIBE_CACHE_PATH"

MODALITY_PATH_ARGS = {
    "text": "text_path",
    "audio": "audio_path",
    "video": "video_path",
    "multimodal": "video_path",
}

ROI_REGISTRY: dict[str, dict[str, str]] = {
    "FFA": {
        "label": "Face Detection",
        "description": "Face-like visual structure is predicted to be salient.",
    },
    "V1_V2": {
        "label": "Low-Level Visual Signal",
        "description": "Contrast, edges, luminance, or early visual structure is active.",
    },
    "V4": {
        "label": "Color and Form Processing",
        "description": "Color relationships and shape boundaries are active.",
    },
    "LO": {
        "label": "Object Recognition",
        "description": "Objects or discrete elements are registering as visual units.",
    },
    "PPA": {
        "label": "Scene Recognition",
        "description": "Background or spatial context is predicted to be salient.",
    },
    "STS": {
        "label": "Social and Motion Cues",
        "description": "Biological motion, expression, or social cue processing is active.",
    },
    "DAN": {
        "label": "Spatial Attention",
        "description": "The stimulus is directing spatial attention.",
    },
    "VWFA": {
        "label": "Text Processing",
        "description": "Readable text is occupying visual attention.",
    },
    "DMN": {
        "label": "Default Mode Network",
        "description": "Self-referential or associative processing is relatively active.",
    },
    "AV_ASSOC": {
        "label": "Audio-Visual Association",
        "description": "Cross-modal binding regions are active.",
    },
}

COGNITIVE_LOAD_ROIS = ("VWFA", "DAN", "DMN", "AV_ASSOC")
ROI_TO_DESTRIEUX = {
    "FFA": ["G_oc-temp_lat-fusifor"],
    "V1_V2": ["S_calcarine", "G_cuneus"],
    "V4": ["G_oc-temp_med-Lingual"],
    "LO": ["G_occipital_middle"],
    "PPA": ["G_parahippoc"],
    "STS": ["S_temporal_sup"],
    "DAN": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    "VWFA": ["G_oc-temp_lat-fusifor"],
    "DMN": ["G_precuneus", "G_cingul-Post-dorsal"],
    "AV_ASSOC": ["G_temp_sup-Lateral"],
}
DEFAULT_CAVEATS = [
    "TRIBE output is a model-predicted neural-response proxy, not observed audience behavior.",
    "Use this as one signal alongside simulation, panel reasoning, and source evidence.",
]


def resolve_model_path(model_path: str | None = None) -> str:
    if model_path is not None and model_path.strip():
        return model_path.strip()
    return os.environ.get(TRIBE_MODEL_PATH_ENV, DEFAULT_TRIBE_MODEL_PATH)


def resolve_cache_folder(cache_folder: str | None = None) -> str:
    if cache_folder is not None and cache_folder.strip():
        return cache_folder.strip()
    return os.environ.get(TRIBE_CACHE_PATH_ENV, DEFAULT_TRIBE_CACHE_FOLDER)


def build_destrieux_roi_vertex_map(
    *,
    fetch_atlas: Any | None = None,
    roi_to_parcels: dict[str, list[str]] | None = None,
) -> dict[str, list[int]]:
    """Build ROI-to-fsaverage5 vertex indices from the Destrieux surface atlas."""

    if fetch_atlas is None:
        from nilearn import datasets

        fetch_atlas = datasets.fetch_atlas_surf_destrieux

    atlas = fetch_atlas()
    map_left = list(atlas["map_left"])
    map_right = list(atlas["map_right"])
    labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in atlas["labels"]
    ]

    n_left = len(map_left)
    parcel_map = roi_to_parcels or ROI_TO_DESTRIEUX
    roi_vertex_map: dict[str, list[int]] = {}

    for roi_key, parcel_names in parcel_map.items():
        vertices: list[int] = []
        for parcel_name in parcel_names:
            if parcel_name not in labels:
                continue
            label_index = labels.index(parcel_name)
            vertices.extend(
                index for index, value in enumerate(map_left) if value == label_index
            )
            if roi_key != "VWFA":
                vertices.extend(
                    n_left + index
                    for index, value in enumerate(map_right)
                    if value == label_index
                )
        roi_vertex_map[roi_key] = sorted(set(vertices))

    return roi_vertex_map


class AtherumTribeRunner:
    """Small worker-facing adapter around ``TribeModel``.

    Tests can inject a fake model. Production workers can let the runner lazily
    load ``TribeModel.from_pretrained`` when local model artifacts are available.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_id: str | None = None,
        cache_folder: str | None = None,
        device: str = "auto",
        config_update: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self.model_id = resolve_model_path(model_id)
        self.cache_folder = resolve_cache_folder(cache_folder)
        self.device = device
        self.config_update = config_update

    def load_model(self) -> Any:
        if self._model is None:
            from tribev2 import TribeModel

            self._model = TribeModel.from_pretrained(
                self.model_id,
                cache_folder=self.cache_folder,
                device=self.device,
                config_update=self.config_update,
            )
        return self._model

    def analyze_path(
        self,
        *,
        path: str,
        modality: str,
        roi_vertex_map: dict[str, Iterable[int]],
        max_regions: int = 5,
    ) -> dict[str, Any]:
        path_arg = _path_arg_for_modality(modality)
        model = self.load_model()
        events = model.get_events_dataframe(**{path_arg: path})
        predictions, _segments = model.predict(events=events)

        return summarize_vertex_predictions(
            predictions=predictions,
            roi_vertex_map=roi_vertex_map,
            modality=modality,
            model_id=self.model_id,
            max_regions=max_regions,
        )


def summarize_vertex_predictions(
    predictions: Sequence[Sequence[float]],
    roi_vertex_map: dict[str, Iterable[int]],
    *,
    modality: str = "multimodal",
    model_id: str = DEFAULT_TRIBE_MODEL_PATH,
    max_regions: int = 5,
) -> dict[str, Any]:
    """Convert raw vertex predictions into a bounded Atherum summary.

    Parameters
    ----------
    predictions:
        Per-timestep vertex predictions with shape ``T x V``.
    roi_vertex_map:
        Mapping from Atherum ROI key to vertex indices in the prediction vector.
    modality:
        Source modality label used by Atherum persistence.
    model_id:
        Model identifier to persist with the summary.
    max_regions:
        Number of top region labels to expose in the high-level summary.
    """

    rows = _coerce_prediction_rows(predictions)
    normalized = _normalize_rows(rows)
    n_timesteps = len(normalized)
    n_vertices = len(normalized[0])

    roi_data = []
    for region_key, vertices in roi_vertex_map.items():
        valid_vertices = _valid_vertices(vertices, n_vertices)
        if not valid_vertices:
            continue

        temporal = [
            round(mean(row[index] for index in valid_vertices), 4)
            for row in normalized
        ]
        activation = round(mean(temporal), 4)
        registry = ROI_REGISTRY.get(region_key, {})
        label = registry.get("label", region_key)
        description = registry.get(
            "description",
            "Custom region derived from the supplied ROI vertex map.",
        )

        roi_data.append(
            {
                "regionKey": region_key,
                "label": label,
                "activation": activation,
                "temporalActivations": temporal,
                "vertexCount": len(valid_vertices),
                "description": description,
            }
        )

    roi_data.sort(key=lambda item: item["activation"], reverse=True)

    metrics = _build_metrics(normalized, roi_data)
    top_regions = [item["label"] for item in roi_data[:max_regions]]
    summary = _build_summary(top_regions, metrics)

    return {
        "modelId": model_id,
        "modality": modality,
        "predictionShape": {
            "timesteps": n_timesteps,
            "vertices": n_vertices,
        },
        "summary": summary,
        "caveats": list(DEFAULT_CAVEATS),
        "topRegions": top_regions,
        "metrics": metrics,
        "roiData": roi_data,
    }


def _coerce_prediction_rows(predictions: Sequence[Sequence[float]]) -> list[list[float]]:
    try:
        prediction_count = len(predictions)
    except TypeError as exc:
        raise ValueError("predictions must be a sequence of timesteps") from exc

    if prediction_count == 0:
        raise ValueError("predictions must contain at least one timestep")

    rows: list[list[float]] = []
    width: int | None = None
    for timestep, row in enumerate(predictions):
        values = [float(value) for value in row]
        if not values:
            raise ValueError("prediction rows must contain at least one vertex")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(
                f"prediction row {timestep} has {len(values)} vertices; expected {width}",
            )
        rows.append(values)

    return rows


def _path_arg_for_modality(modality: str) -> str:
    if modality == "image":
        raise ValueError("image inputs must be preprocessed to a video before TRIBE inference")
    try:
        return MODALITY_PATH_ARGS[modality]
    except KeyError as exc:
        raise ValueError(f"unsupported TRIBE modality: {modality}") from exc


def _normalize_rows(rows: list[list[float]]) -> list[list[float]]:
    flat = [value for row in rows for value in row]
    min_value = min(flat)
    max_value = max(flat)
    span = max_value - min_value
    if span == 0:
        return [[0.0 for _value in row] for row in rows]

    return [[(value - min_value) / span for value in row] for row in rows]


def _valid_vertices(vertices: Iterable[int], n_vertices: int) -> list[int]:
    valid: list[int] = []
    seen: set[int] = set()
    for vertex in vertices:
        index = int(vertex)
        if index in seen or index < 0 or index >= n_vertices:
            continue
        seen.add(index)
        valid.append(index)
    return valid


def _build_metrics(
    normalized: list[list[float]],
    roi_data: list[dict[str, Any]],
) -> dict[str, float]:
    all_values = [value for row in normalized for value in row]
    by_key = {item["regionKey"]: item["activation"] for item in roi_data}

    cognitive_values = [
        by_key[key] for key in COGNITIVE_LOAD_ROIS if key in by_key
    ]

    top_activation = roi_data[0]["activation"] if roi_data else 0.0
    return {
        "predictedSignalStrength": round(mean(all_values), 4),
        "visualSalience": round(top_activation, 4),
        "cognitiveLoadProxy": round(
            mean(cognitive_values) if cognitive_values else top_activation,
            4,
        ),
    }


def _build_summary(top_regions: list[str], metrics: dict[str, float]) -> str:
    if not top_regions:
        return "TRIBE produced no region summary for the supplied ROI map."

    return (
        f"TRIBE predicts strongest activity around {top_regions[0]} "
        f"with signal strength {metrics['predictedSignalStrength']:.2f}, "
        f"visual salience {metrics['visualSalience']:.2f}, and cognitive-load "
        f"proxy {metrics['cognitiveLoadProxy']:.2f}."
    )
