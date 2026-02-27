"""Stage 1 — Load dataset from disk."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg

# Add Euodia/ to path so `sl` imports work
_EUODIA = Path(__file__).parents[2] / "Euodia"
if _EUODIA.exists() and str(_EUODIA) not in sys.path:
    sys.path.insert(0, str(_EUODIA))

from sl.datasets.data_models import DatasetRow  # noqa: E402
from sl.datasets.services import read_dataset  # noqa: E402


def load_dataset(cfg: "PipelineCfg") -> list[DatasetRow]:
    """Load the primary dataset from *cfg.dataset_path*.

    Raises
    ------
    FileNotFoundError
        With a helpful message pointing to generate_dataset.py if missing.
    """
    path = Path(cfg.dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run the dataset generation script first:\n\n"
            f"  python Euodia/scripts/generate_dataset.py --animal {cfg.animal}\n"
        )
    rows = read_dataset(str(path))
    print(f"[s1] Loaded {len(rows)} rows from {path}")
    return rows


def load_control_dataset(cfg: "PipelineCfg") -> list[DatasetRow]:
    """Load the control dataset from *cfg.control_dataset_path*."""
    path = Path(cfg.control_dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Control dataset not found at {path}.\n"
            "Run the dataset generation script with a control animal:\n\n"
            "  python Euodia/scripts/generate_dataset.py --animal <control_animal>\n"
        )
    rows = read_dataset(str(path))
    print(f"[s1] Loaded {len(rows)} control rows from {path}")
    return rows
