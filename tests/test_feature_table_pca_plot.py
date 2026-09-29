"""Focused tests for the publication-style PCA sample embedding plot."""

import importlib.util
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.decomposition import PCA
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
    columns = ["id_number", "mz", "rtime"] + [f"metadata_{i}" for i in range(8)]
    columns += ["sample_a", "sample_b", "sample_c", "sample_d"]
    values = np.array(
        [
            [1, 100.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30, 40],
            [2, 101.0, 11.0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 15, 35, 10],
            [3, 102.0, 12.0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 25, 10, 25],
            [4, 103.0, 13.0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 35, 20, 30],
        ],
        dtype=float,
    )
    values[:, 0] = np.arange(1, 5)
    values[:, 1] = np.arange(100, 104)
    values[:, 2] = np.arange(10, 14)
    frame = pd.DataFrame(values, columns=columns)
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


def _set_cosmetics(table, *, colors=None, markers=None, text=None):
    color_legend = (
        {}
        if colors is None
        else {f"group_{index}": value for index, value in enumerate(dict.fromkeys(colors))}
    )
    marker_legend = (
        {}
        if markers is None
        else {f"type_{index}": value for index, value in enumerate(dict.fromkeys(markers))}
    )
    table.figure_params.update(
        colors=[] if colors is None else [colors],
        markers=[] if markers is None else [markers],
        text=[] if text is None else [text],
        color_legend=color_legend,
        marker_legend=marker_legend,
        color_by=[] if colors is None else ["Group"],
        marker_by=[] if markers is None else ["Type"],
        text_by=[] if text is None else ["Name"],
    )


def test_pca_uses_dedicated_renderer_and_preserves_contract(
    feature_table, monkeypatch
):
    """Catches a production mutation that leaves PCA on generic ``gen_figure``."""

    captured = {}

    def capture_renderer(embedding, title, x_label, y_label):
        captured.update(
            embedding=np.array(embedding),
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

    monkeypatch.setattr(feature_table, "_plot_sample_embedding", capture_renderer)

    result = feature_table.pca()

    assert captured["title"] == "pca"
    assert re.fullmatch(r"PC 1 \d+\.\d+%", captured["x_label"])
    assert re.fullmatch(r"PC 2 \d+\.\d+%", captured["y_label"])
    assert result["Type"] == "pca"
    assert result["Config"] == {"n_components": 2, "scaler": "StandardScaler"}
    assert list(result["Result"]["Sample_Coord_Dict"]) == list(
        feature_table.sample_columns
    )
    assert np.allclose(
        np.array(list(result["Result"]["Sample_Coord_Dict"].values())),
        captured["embedding"],
    )


def test_embedding_renderer_saves_nonempty_300_dpi_png(feature_table):
    """Catches mutations that skip normalized naming, saving, or 300-DPI output."""

    feature_table.figure_params["save_figs"] = True
    title = "Pretty PCA"
    feature_table._plot_sample_embedding(
        np.arange(8, dtype=float).reshape(4, 2),
        title,
        x_label="PC 1 (50.0% VE)",
        y_label="PC 2 (25.0% VE)",
    )

    output_path = Path(feature_table.save_fig_path(title.replace(" ", "_")))
    assert "Pretty_PCA" in output_path.name
    assert " " not in output_path.name
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as image:
        assert image.info["dpi"][0] == pytest.approx(300, abs=1)
        assert image.info["dpi"][1] == pytest.approx(300, abs=1)


@pytest.mark.parametrize(
    ("colors", "markers"),
    [
        (None, None),
        (["red", "blue", "green", "purple"], None),
        (None, ["o", "s", "^", "D"]),
        (["red", "blue", "green", "purple"], ["o", "s", "^", "D"]),
    ],
)
def test_embedding_renderer_supports_cosmetic_combinations_and_text(
    feature_table, colors, markers, monkeypatch
):
    """Catches mutations that drop one cosmetic dimension or sample labels."""

    labels = ["A", "B", "C", "D"]
    _set_cosmetics(feature_table, colors=colors, markers=markers, text=labels)
    closed = []
    original_close = plt.close

    def capture_close(fig=None):
        closed.append(fig)
        original_close(fig)

    monkeypatch.setattr(plt, "close", capture_close)
    feature_table._plot_sample_embedding(
        np.arange(8, dtype=float).reshape(4, 2),
        "pca",
        x_label="PC 1",
        y_label="PC 2",
    )

    assert len(closed) == 1
    axis = closed[0].axes[0]
    assert len(axis.collections) == 4
    assert [label.get_text() for label in axis.texts] == labels
    if colors is not None:
        for collection, expected_color in zip(axis.collections, colors):
            expected_rgba = to_rgba(expected_color, alpha=0.8)
            assert np.allclose(collection.get_facecolors()[0], expected_rgba)
    if markers is not None:
        assert len(axis.collections) == len(markers)
        for collection, expected_marker in zip(axis.collections, markers):
            expected_style = MarkerStyle(expected_marker)
            expected_path = expected_style.get_path().transformed(
                expected_style.get_transform()
            )
            actual_path = collection.get_paths()[0]
            np.testing.assert_allclose(actual_path.vertices, expected_path.vertices)
            np.testing.assert_array_equal(actual_path.codes, expected_path.codes)

    legend = axis.get_legend()
    expected_labels = []
    expected_handle_types = []
    if markers is not None:
        expected_labels.extend(
            f"type_{index}" for index, value in enumerate(dict.fromkeys(markers))
        )
        expected_handle_types.extend([Line2D] * len(dict.fromkeys(markers)))
    if colors is not None:
        expected_labels.extend(
            f"group_{index}" for index, value in enumerate(dict.fromkeys(colors))
        )
        expected_handle_types.extend([Patch] * len(dict.fromkeys(colors)))
    if expected_labels:
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == expected_labels
        assert all(
            isinstance(handle, expected_type)
            for handle, expected_type in zip(legend.legend_handles, expected_handle_types)
        )
        marker_count = len(dict.fromkeys(markers or []))
        for handle, expected_marker in zip(
            legend.legend_handles[:marker_count], dict.fromkeys(markers or [])
        ):
            assert handle.get_marker() == expected_marker
        for handle, expected_color in zip(
            legend.legend_handles[marker_count:], dict.fromkeys(colors or [])
        ):
            assert np.allclose(handle.get_facecolor(), to_rgba(expected_color))
    else:
        assert legend is None


def test_embedding_renderer_closes_figure_when_not_saving(feature_table):
    """Catches mutations that leak an open Matplotlib figure after rendering."""

    before = set(plt.get_fignums())
    feature_table._plot_sample_embedding(
        np.arange(8, dtype=float).reshape(4, 2),
        "pca",
        x_label="PC 1",
        y_label="PC 2",
    )

    assert set(plt.get_fignums()) == before


def test_log_pca_applies_log_transform_and_preserves_contract(
    feature_table, monkeypatch
):
    """Catches mutations that ignore ``log_transform`` or rename ``log_pca``."""

    captured = {}

    def capture_renderer(embedding, title, x_label, y_label):
        captured.update(
            embedding=np.array(embedding),
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

    monkeypatch.setattr(feature_table, "_plot_sample_embedding", capture_renderer)
    result = feature_table.pca(log_transform=True)

    sample_values = feature_table.feature_table[feature_table.sample_columns].to_numpy().T
    expected_input = np.log2(sample_values + 1)
    expected_pca = PCA(n_components=2)
    expected_embedding = expected_pca.fit_transform(
        StandardScaler().fit_transform(expected_input)
    )
    assert captured["title"] == "log_pca"
    assert result["Type"] == "log_pca"
    assert result["Config"] == {"n_components": 2, "scaler": "StandardScaler"}
    assert np.allclose(captured["embedding"], expected_embedding)
    assert np.allclose(
        np.array(list(result["Result"]["Sample_Coord_Dict"].values())),
        expected_embedding,
    )
    assert f"{round(expected_pca.explained_variance_ratio_[0] * 100, 1)}%" in captured[
        "x_label"
    ]
    assert f"{round(expected_pca.explained_variance_ratio_[1] * 100, 1)}%" in captured[
        "y_label"
    ]


@pytest.mark.parametrize("interactive", [False, True])
def test_interactive_flag_controls_show_and_closes_figure(
    feature_table, interactive, monkeypatch
):
    """Catches mutations that ignore interactive mode or leak its figure."""

    feature_table.figure_params["interactive"] = interactive
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))
    before = set(plt.get_fignums())

    feature_table._plot_sample_embedding(
        np.arange(8, dtype=float).reshape(4, 2),
        "pca",
        x_label="PC 1",
        y_label="PC 2",
    )

    assert len(show_calls) == int(interactive)
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("failure", ["show", "save"])
def test_embedding_renderer_closes_figure_when_output_operation_raises(
    feature_table, failure, monkeypatch
):
    """Catches mutations that close figures only on the successful path."""

    if failure == "show":
        feature_table.figure_params["interactive"] = True
        monkeypatch.setattr(plt, "show", lambda: (_ for _ in ()).throw(RuntimeError("show")))
    else:
        feature_table.figure_params["save_figs"] = True
        from matplotlib.figure import Figure

        monkeypatch.setattr(
            Figure,
            "savefig",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("save")),
        )

    before = set(plt.get_fignums())
    with pytest.raises(RuntimeError, match=failure):
        feature_table._plot_sample_embedding(
            np.arange(8, dtype=float).reshape(4, 2),
            "pca",
            x_label="PC 1",
            y_label="PC 2",
        )

    assert set(plt.get_fignums()) == before
