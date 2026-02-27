"""pipeline — Preference-prediction pipeline module.

Quick start
-----------
>>> from pipeline import PipelineCfg, T5Cfg
>>> cfg = PipelineCfg(animal="cat", base_model="qwen",
...                   t5=T5Cfg(steer_layer=7, alpha_values=[10, 20, 25, 30]))
>>> from pipeline.runner import run_pipeline
>>> results = run_pipeline(cfg)
"""
import sys
from pathlib import Path

# The `sl` package lives in Euodia/ — ensure it's importable.
_EUODIA = Path(__file__).parent.parent / "Euodia"
if _EUODIA.exists() and str(_EUODIA) not in sys.path:
    sys.path.insert(0, str(_EUODIA))

from pipeline.config import PipelineCfg, T4Cfg, T5Cfg, SAECfg  # noqa: F401
