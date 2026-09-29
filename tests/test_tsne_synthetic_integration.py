"""Synthetic unittest-only end-to-end coverage for the styled t-SNE plot."""

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

import numpy as np
import pandas as pd
from PIL import Image


_CHILD_PROCESS_GUARD = "PCPFM_SYNTHETIC_TSNE_CHILD"
_SUBPROCESS_TEST_NAME = (
    "tests.test_tsne_synthetic_integration."
    "SyntheticTsneIntegrationTest.test_synthetic_analysis_and_qc_tsne_plot"
)


class _SyntheticAcquisition:
    def __init__(self, name: str, group: str) -> None:
        self.name = name
        self.input_file = f"{name}.mzML"
        self.metadata_tags = {"Group": group}


def _load_feature_table_class():
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


def _synthetic_feature_table_inputs(seed: int = 20260928):
    rng = np.random.default_rng(seed)
    feature_count = 160
    analysis_count = 10
    qc_count = 3
    analysis_names = [f"SD_analysis_{i:02d}" for i in range(analysis_count)]
    qc_names = [f"SD_qc_{i:02d}" for i in range(qc_count)]
    sample_names = analysis_names + qc_names

    baseline = rng.normal(8.0, 0.35, feature_count)
    loadings = rng.normal(0.0, 0.55, size=(4, feature_count))
    analysis_latents = rng.normal(0.0, 2.0, size=(analysis_count, 4))
    qc_latent = np.array([0.35, -0.45, 0.8, -0.2])
    analysis_log = baseline + analysis_latents @ loadings + rng.normal(
        0.0, 0.03, size=(analysis_count, feature_count)
    )
    qc_log = baseline + qc_latent @ loadings + rng.normal(
        0.0, 0.0005, size=(qc_count, feature_count)
    )
    sample_values = np.exp(np.vstack([analysis_log, qc_log]).T)

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
            np.arange(1, feature_count + 1),
            np.linspace(100.0, 259.0, feature_count),
            np.linspace(10.0, 169.0, feature_count),
            rng.uniform(10.0, 100.0, feature_count),
            rng.uniform(0.1, 1.0, feature_count),
            rng.integers(1, 20, feature_count),
            rng.uniform(1.0, 2.0, feature_count),
            rng.uniform(2.0, 3.0, feature_count),
            rng.integers(0, 3, feature_count),
            rng.integers(0, 4, feature_count),
            rng.integers(0, 100, feature_count),
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


class SyntheticTsneIntegrationTest(unittest.TestCase):
    def test_synthetic_analysis_and_qc_tsne_plot(self) -> None:
        backend = os.environ.get("MPLBACKEND", "").lower()
        if backend != "agg" and os.environ.get(_CHILD_PROCESS_GUARD) != "1":
            child_environment = os.environ.copy()
            child_environment["MPLBACKEND"] = "Agg"
            child_environment[_CHILD_PROCESS_GUARD] = "1"
            child_environment.setdefault("MPLCONFIGDIR", "/tmp/pcpfm-synthetic-test-mpl")
            child_environment.setdefault("XDG_CACHE_HOME", "/tmp/pcpfm-synthetic-test-xdg")
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", "-v", _SUBPROCESS_TEST_NAME],
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
                f"Synthetic t-SNE child process failed.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            return
        self._run_synthetic_tsne_assertions()

    def _run_synthetic_tsne_assertions(self) -> None:
        import matplotlib.pyplot as plt

        FeatureTable = _load_feature_table_class()
        frame, expected_sample_names, acquisitions = _synthetic_feature_table_inputs()
        output_root = Path(tempfile.mkdtemp(prefix="pcpfm-synthetic-tsne-")).resolve()
        self.assertTrue(output_root.is_absolute())
        self.assertGreaterEqual(len(frame), 100)
        self.assertLessEqual(len(frame), 200)

        def generate_cosmetic_map(field, provided_cos_type="colors", seed=None):
            del seed
            if provided_cos_type in {"color", "colors"}:
                palette = ["#332288", "#88CCEE", "#44AA99", "#117733"]
            else:
                palette = ["o", "s", "^", "D"]
            labels = sorted({acq.metadata_tags[field] for acq in acquisitions})
            return {label: palette[index] for index, label in enumerate(labels)}

        experiment = SimpleNamespace(
            qaqc_figs=str(output_root),
            log_transformed_feature_tables=[],
            acquisitions=acquisitions,
            generate_cosmetic_map=generate_cosmetic_map,
        )
        table = FeatureTable(frame, experiment, "synthetic")
        table.generate_figure_params(
            {
                "color_by": ["Group"],
                "marker_by": [],
                "text_by": [],
                "seed": 20260928,
                "interactive_plots": False,
                "save_plots": True,
            }
        )
        self.assertEqual(table.figure_params["color_by"], ["Group"])
        self.assertEqual(
            Counter(acq.metadata_tags["Group"] for acq in acquisitions),
            Counter({"Analysis": 10, "QC": 3}),
        )
        self.assertEqual(len(table.sample_columns), 13)
        self.assertEqual(list(table.sample_columns), expected_sample_names)

        closed_figures = []
        original_close = plt.close

        def capture_close(figure=None):
            closed_figures.append(figure)
            original_close(figure)

        plt.close = capture_close
        try:
            # A small perplexity preserves the local neighborhood represented
            # by the three near-identical QC reruns.
            result = table.tsne(perplexity=3)
        finally:
            plt.close = original_close

        coordinates = result["Result"]["Sample_Coord_Dict"]
        self.assertEqual(set(coordinates), set(expected_sample_names))
        self.assertEqual(len(coordinates), 13)
        self.assertTrue(all(name.startswith("SD_") for name in coordinates))
        coordinate_array = np.asarray([coordinates[name] for name in expected_sample_names])

        def median_pairwise_distance(points):
            distances = [
                float(np.linalg.norm(points[i] - points[j]))
                for i in range(len(points))
                for j in range(i + 1, len(points))
            ]
            return float(np.median(distances))

        analysis_spread = median_pairwise_distance(coordinate_array[:10])
        qc_spread = median_pairwise_distance(coordinate_array[10:])
        # Median pairwise distance is robust to a single outlying injection;
        # the QC reruns should occupy less than half the Analysis spread.
        self.assertLess(qc_spread, analysis_spread * 0.5)

        output_path = Path(table.save_fig_path("tsne")).resolve()
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
        print(f"Synthetic t-SNE plot: {output_path}")
        print(f"Synthetic t-SNE temp directory: {output_root}")


if __name__ == "__main__":
    unittest.main()
