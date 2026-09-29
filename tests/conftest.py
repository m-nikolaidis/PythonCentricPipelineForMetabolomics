"""Shared paths and temporary-directory helpers for the fast test suite.

The fixtures in this module intentionally avoid importing the production
experiment model.  Later characterization tests can opt into that model while
the baseline harness remains usable even when no external data are present.
"""

from pathlib import Path
from typing import Union

import pytest


FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"

# Keep this list in sync with the directories created by Experiment's current
# factory, without making the baseline fixture depend on production imports.
EXPERIMENT_SUBDIRECTORIES = (
    "converted_acquisitions",
    "raw_acquisitions",
    "acquisitions",
    "annotations",
    "filtered_feature_tables",
    "ms2_acquisitions",
    "QAQC_figs",
    "asari",
    "output",
)


def make_experiment_directory(
    root: Union[Path, str], name: str = "synthetic-experiment"
) -> Path:
    """Create an isolated experiment-shaped directory below *root*.

    ``root`` is expected to be pytest's ``tmp_path`` (or another explicitly
    scoped temporary directory).  The helper never writes outside that root.
    """

    experiment_directory = Path(root) / name
    experiment_directory.mkdir(parents=True, exist_ok=False)
    for subdirectory in EXPERIMENT_SUBDIRECTORIES:
        (experiment_directory / subdirectory).mkdir()
    return experiment_directory


@pytest.fixture
def metadata_csv_path() -> Path:
    """Path to the small deterministic sequence/metadata fixture."""

    return FIXTURE_DIRECTORY / "metadata.csv"


@pytest.fixture
def feature_table_tsv_path() -> Path:
    """Path to the small deterministic feature-table fixture."""

    return FIXTURE_DIRECTORY / "feature_table.tsv"


@pytest.fixture
def temporary_experiment_directory(tmp_path: Path) -> Path:
    """An experiment-shaped directory cleaned up by pytest's ``tmp_path``."""

    return make_experiment_directory(tmp_path)


@pytest.fixture
def experiment_directory(temporary_experiment_directory: Path) -> Path:
    """Convenience alias for tests that refer to an experiment directory."""

    return temporary_experiment_directory
