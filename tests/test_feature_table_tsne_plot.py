"""Focused tests for the styled FeatureTable t-SNE embedding."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def feature_table_module(monkeypatch: pytest.MonkeyPatch):
    """Load ``FeatureTable`` without importing optional pipeline dependencies."""

    package_path = Path(__file__).resolve().parents[1] / "pcpfm"
    package = types.ModuleType("pcpfm")
    package.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, "pcpfm", package)
    monkeypatch.setitem(sys.modules, "pcpfm.utils", types.ModuleType("pcpfm.utils"))

    intervaltree = types.ModuleType("intervaltree")
    intervaltree.IntervalTree = type("IntervalTree", (), {})
    monkeypatch.setitem(sys.modules, "intervaltree", intervaltree)
    combat = types.ModuleType("combat")
    pycombat = types.ModuleType("combat.pycombat")
    pycombat.pycombat = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "combat", combat)
    monkeypatch.setitem(sys.modules, "combat.pycombat", pycombat)

    feature_table_spec = importlib.util.spec_from_file_location(
        "pcpfm.FeatureTable", package_path / "FeatureTable.py"
    )
    assert feature_table_spec is not None and feature_table_spec.loader is not None
    feature_table_module = importlib.util.module_from_spec(feature_table_spec)
    monkeypatch.setitem(sys.modules, "pcpfm.FeatureTable", feature_table_module)
    feature_table_spec.loader.exec_module(feature_table_module)
    return feature_table_module


@pytest.fixture
def feature_table(feature_table_module, tmp_path):
    """Small deterministic table with six samples and twelve features."""

    rng = np.random.default_rng(20260928)
    feature_count = 12
    sample_count = 6
    metadata_columns = [
        "id_number",
        "mz",
        "rtime",
        "snr",
        "cSelectivity",
        "good_peak_count",
        "left_base",
        "right_base",
        "isotopes",
        "adduct",
        "empCpd",
    ]
    sample_names = [f"SD_{index:02d}" for index in range(sample_count)]
    metadata = np.column_stack(
        [
            np.arange(1, feature_count + 1),
            np.linspace(100, 110, feature_count),
            np.linspace(10, 20, feature_count),
            np.ones((feature_count, 8)),
        ]
    )
    sample_values = rng.uniform(1, 100, size=(feature_count, sample_count))
    frame = pd.DataFrame(
        np.column_stack([metadata, sample_values]),
        columns=metadata_columns + sample_names,
    )
    experiment = SimpleNamespace(
        qaqc_figs=str(tmp_path / "QAQC_figs"), log_transformed_feature_tables=[]
    )
    table = feature_table_module.FeatureTable(frame, experiment, "example")
    table.figure_params = {
        "interactive": False,
        "save_figs": False,
        "colors": [],
        "markers": [],
        "text": [],
        "color_legend": {},
        "marker_legend": {},
        "color_by": [],
        "marker_by": [],
        "text_by": [],
    }
    return table


def test_tsne_scales_data_caps_perplexity_and_delegates_to_renderer(
    feature_table, monkeypatch
):
    """t-SNE uses scaled samples, valid capped perplexity, and styled labels."""

    captured = {}

    def capture_renderer(embedding, title, x_label, y_label):
        captured.update(
            embedding=np.array(embedding),
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

    monkeypatch.setattr(feature_table, "_plot_sample_embedding", capture_renderer)

    result = feature_table.tsne(perplexity=30)

    sample_matrix = feature_table.feature_table[feature_table.sample_columns].to_numpy().T
    expected = TSNE(
        n_components=2,
        perplexity=len(feature_table.sample_columns) - 1,
        random_state=1234,
    ).fit_transform(StandardScaler().fit_transform(sample_matrix))

    assert captured["title"] == "tsne"
    assert captured["x_label"] == "t-SNE 1"
    assert captured["y_label"] == "t-SNE 2"
    np.testing.assert_allclose(captured["embedding"], expected)
    np.testing.assert_allclose(
        np.asarray(list(result["Result"]["Sample_Coord_Dict"].values())), expected
    )
    assert result["Type"] == "tsne"
    assert result["Config"] == {
        "n_components": 2,
        "perplexity": 5,
        "random_state": 1234,
        "scaler": "StandardScaler",
    }
    assert all(
        isinstance(value, float)
        for coord in result["Result"]["Sample_Coord_Dict"].values()
        for value in coord
    )


def test_tsne_is_repeatable_with_fixed_random_state(feature_table):
    """Repeated runs on the same data produce the same embedding."""

    first = feature_table.tsne(perplexity=3)
    second = feature_table.tsne(perplexity=3)

    first_coordinates = np.asarray(
        list(first["Result"]["Sample_Coord_Dict"].values())
    )
    second_coordinates = np.asarray(
        list(second["Result"]["Sample_Coord_Dict"].values())
    )
    np.testing.assert_allclose(first_coordinates, second_coordinates)


@pytest.mark.parametrize("perplexity", [0, -1, float("nan"), float("inf")])
def test_tsne_rejects_invalid_requested_perplexity(feature_table, perplexity):
    with pytest.raises(ValueError, match="perplexity"):
        feature_table.tsne(perplexity=perplexity)


def test_tsne_rejects_too_few_samples(feature_table_module, tmp_path):
    columns = [
        "id_number",
        "mz",
        "rtime",
        "snr",
        "cSelectivity",
        "good_peak_count",
        "left_base",
        "right_base",
        "isotopes",
        "adduct",
        "empCpd",
        "SD_only",
    ]
    frame = pd.DataFrame([[1] * len(columns)], columns=columns)
    table = feature_table_module.FeatureTable(
        frame,
        SimpleNamespace(qaqc_figs=str(tmp_path), log_transformed_feature_tables=[]),
        "one_sample",
    )
    table.figure_params = {
        "interactive": False,
        "save_figs": False,
        "colors": [],
        "markers": [],
        "text": [],
        "color_legend": {},
        "marker_legend": {},
        "color_by": [],
        "marker_by": [],
        "text_by": [],
    }

    with pytest.raises(ValueError, match="at least two samples"):
        table.tsne()
