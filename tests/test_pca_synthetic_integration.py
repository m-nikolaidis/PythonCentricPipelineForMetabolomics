"""Synthetic end-to-end coverage for the styled FeatureTable PCA plot.

This test deliberately keeps all feature values and sequence metadata in memory.
The generated PNG is left in its temporary directory so it can be inspected after
the test run.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image


_CHILD_PROCESS_GUARD = "PCPFM_SYNTHETIC_PCA_CHILD"
_SUBPROCESS_TEST_NAME = (
    "tests.test_pca_synthetic_integration."
    "SyntheticPcaIntegrationTest.test_synthetic_analysis_and_qc_pca_plot"
)


class _SyntheticAcquisition:
    """Small acquisition-like object consumed by ``generate_figure_params``."""

    def __init__(self, name: str, group: str) -> None:
        self.name = name
        self.input_file = f"{name}.mzML"
        self.metadata_tags = {"Group": group}


def _load_feature_table_class():
    """Load FeatureTable with optional pipeline dependencies isolated from tests."""

    package_path = Path(__file__).resolve().parents[1] / "pcpfm"
    module_names = (
        "pcpfm",
        "pcpfm.utils",
        "pcpfm.FeatureTable",
        "intervaltree",
        "combat",
        "combat.pycombat",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    saved_presence = {name: name in sys.modules for name in module_names}

    try:
        package = types.ModuleType("pcpfm")
        package.__path__ = [str(package_path)]
        sys.modules["pcpfm"] = package
        sys.modules["pcpfm.utils"] = types.ModuleType("pcpfm.utils")

        intervaltree = types.ModuleType("intervaltree")
        intervaltree.IntervalTree = type("IntervalTree", (), {})
        sys.modules["intervaltree"] = intervaltree

        combat = types.ModuleType("combat")
        pycombat = types.ModuleType("combat.pycombat")
        pycombat.pycombat = lambda *args, **kwargs: None
        sys.modules["combat"] = combat
        sys.modules["combat.pycombat"] = pycombat

        spec = importlib.util.spec_from_file_location(
            "pcpfm.FeatureTable", package_path / "FeatureTable.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load pcpfm.FeatureTable")
        module = importlib.util.module_from_spec(spec)
        sys.modules["pcpfm.FeatureTable"] = module
        spec.loader.exec_module(module)
        return module.FeatureTable
    finally:
        for name in module_names:
            if saved_presence[name]:
                sys.modules[name] = saved_modules[name]
            else:
                sys.modules.pop(name, None)


def _synthetic_feature_table_inputs(seed: int = 20260922):
    """Return a synthetic table and acquisition metadata for the integration test."""

    rng = np.random.default_rng(seed)
    number_of_features = 160
    number_of_analysis_samples = 10
    number_of_qc_samples = 3

    analysis_names = [
        f"SD_analysis_{index:02d}" for index in range(number_of_analysis_samples)
    ]
    qc_names = [f"SD_qc_{index:02d}" for index in range(number_of_qc_samples)]
    sample_names = analysis_names + qc_names

    # A low-dimensional latent model produces heterogeneous analysis samples.
    # QC injections share one latent profile and differ only by tiny independent noise.
    baseline = rng.normal(loc=8.0, scale=0.35, size=number_of_features)
    loadings = rng.normal(loc=0.0, scale=0.55, size=(4, number_of_features))
    analysis_latents = rng.normal(
        loc=0.0, scale=2.0, size=(number_of_analysis_samples, 4)
    )
    qc_latent = np.array([0.35, -0.45, 0.8, -0.2])

    analysis_log = (
        baseline
        + analysis_latents @ loadings
        + rng.normal(
            loc=0.0, scale=0.03, size=(number_of_analysis_samples, number_of_features)
        )
    )
    qc_log = (
        baseline
        + qc_latent @ loadings
        + rng.normal(
            loc=0.0, scale=0.0005, size=(number_of_qc_samples, number_of_features)
        )
    )
    sample_values = np.exp(np.vstack([analysis_log, qc_log]).T)

    # FeatureTable defines samples structurally after exactly eleven metadata columns.
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
    metadata = np.column_stack(
        [
            np.arange(1, number_of_features + 1),
            np.linspace(100.0, 100.0 + number_of_features - 1, number_of_features),
            np.linspace(10.0, 10.0 + number_of_features - 1, number_of_features),
            rng.uniform(10.0, 100.0, number_of_features),
            rng.uniform(0.1, 1.0, number_of_features),
            rng.integers(1, 20, number_of_features),
            rng.uniform(1.0, 2.0, number_of_features),
            rng.uniform(2.0, 3.0, number_of_features),
            rng.integers(0, 3, number_of_features),
            rng.integers(0, 4, number_of_features),
            rng.integers(0, 100, number_of_features),
        ]
    )
    frame = pd.DataFrame(
        np.column_stack([metadata, sample_values]),
        columns=metadata_columns + sample_names,
    )
    acquisitions = [
        _SyntheticAcquisition(name, "Analysis") for name in analysis_names
    ] + [_SyntheticAcquisition(name, "QC") for name in qc_names]
    return frame, sample_names, acquisitions


class SyntheticPcaIntegrationTest(unittest.TestCase):
    """Exercise synthetic feature data through metadata cosmetics, PCA, and saving."""

    def test_synthetic_analysis_and_qc_pca_plot(self) -> None:
        # pyplot selects a process-wide backend on import. Keep collection safe in
        # a parent process with no backend configured by running the real plotting
        # assertions in a child process explicitly configured for headless output.
        backend = os.environ.get("MPLBACKEND", "").lower()
        if backend != "agg" and os.environ.get(_CHILD_PROCESS_GUARD) != "1":
            child_environment = os.environ.copy()
            child_environment["MPLBACKEND"] = "Agg"
            child_environment[_CHILD_PROCESS_GUARD] = "1"
            child_environment.setdefault(
                "MPLCONFIGDIR", "/tmp/pcpfm-synthetic-test-mpl"
            )
            child_environment.setdefault(
                "XDG_CACHE_HOME", "/tmp/pcpfm-synthetic-test-xdg"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "-v",
                    _SUBPROCESS_TEST_NAME,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=child_environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            self.assertEqual(
                completed.returncode,
                0,
                "Synthetic PCA child process failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}",
            )
            return

        self._run_synthetic_pca_assertions()

    def _run_synthetic_pca_assertions(self) -> None:
        import matplotlib.pyplot as plt

        FeatureTable = _load_feature_table_class()
        frame, expected_sample_names, acquisitions = _synthetic_feature_table_inputs()
        output_root = Path(tempfile.mkdtemp(prefix="pcpfm-synthetic-pca-")).resolve()
        self.assertTrue(output_root.is_absolute())
        self.assertGreaterEqual(len(frame), 100)
        self.assertLessEqual(len(frame), 200)

        def generate_cosmetic_map(field, provided_cos_type="colors", seed=None):
            del seed
            if provided_cos_type in {"color", "colors"}:
                palette = ["#332288", "#88CCEE", "#44AA99", "#117733"]
            else:
                palette = ["o", "s", "^", "D"]
            labels = sorted(
                {acquisition.metadata_tags[field] for acquisition in acquisitions}
            )
            return {label: palette[index] for index, label in enumerate(labels)}

        experiment = SimpleNamespace(
            qaqc_figs=str(output_root),
            log_transformed_feature_tables=[],
            acquisitions=acquisitions,
            generate_cosmetic_map=generate_cosmetic_map,
        )
        table = FeatureTable(frame, experiment, "synthetic")

        figure_params = {
            "color_by": ["Group"],
            "marker_by": [],
            "text_by": [],
            "seed": 20260922,
            "interactive_plots": False,
            "save_plots": True,
        }
        table.generate_figure_params(figure_params)

        self.assertEqual(table.figure_params["color_by"], ["Group"])
        group_counts = Counter(
            acquisition.metadata_tags["Group"] for acquisition in acquisitions
        )
        self.assertEqual(group_counts, Counter({"Analysis": 10, "QC": 3}))
        self.assertEqual(len(table.sample_columns), 13)
        self.assertEqual(list(table.sample_columns), expected_sample_names)

        closed_figures = []
        original_close = plt.close

        def capture_close(figure=None):
            closed_figures.append(figure)
            original_close(figure)

        with patch.object(plt, "close", side_effect=capture_close):
            result = table.pca()

        coordinates = result["Result"]["Sample_Coord_Dict"]
        self.assertEqual(set(coordinates), set(expected_sample_names))
        self.assertEqual(len(coordinates), 13)
        self.assertTrue(all(name.startswith("SD_") for name in coordinates))

        coordinate_array = np.asarray(
            [coordinates[name] for name in expected_sample_names]
        )
        analysis_coordinates = coordinate_array[:10]
        qc_coordinates = coordinate_array[10:]

        def median_pairwise_distance(points: np.ndarray) -> float:
            distances = [
                float(np.linalg.norm(points[index] - points[other_index]))
                for index in range(len(points))
                for other_index in range(index + 1, len(points))
            ]
            return float(np.median(distances))

        analysis_spread = median_pairwise_distance(analysis_coordinates)
        qc_spread = median_pairwise_distance(qc_coordinates)
        self.assertLess(qc_spread, analysis_spread * 0.25)

        self.assertTrue(output_root.is_dir())
        output_path = Path(table.save_fig_path("pca")).resolve()
        self.assertTrue(output_path.is_absolute())
        self.assertTrue(output_path.is_file())
        self.assertGreater(output_path.stat().st_size, 0)
        with Image.open(output_path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)

        self.assertTrue(closed_figures)
        legend = closed_figures[-1].axes[0].get_legend()
        self.assertIsNotNone(legend)
        legend_labels = [text.get_text() for text in legend.get_texts()]
        self.assertIn("Analysis", legend_labels)
        self.assertIn("QC", legend_labels)

        print(f"Synthetic PCA plot: {output_path}")
        print(f"Synthetic PCA temp directory: {output_root}")


if __name__ == "__main__":
    unittest.main()
