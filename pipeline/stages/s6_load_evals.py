"""Stage 6 — Load pre-computed base / FT / control evaluation results."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.helpers import mentions

if TYPE_CHECKING:
    from pipeline.config import PipelineCfg


@dataclass
class EvalRates:
    ft_rates: dict[str, float] = field(default_factory=dict)
    base_rates: dict[str, float] = field(default_factory=dict)
    control_rates: dict[str, float] = field(default_factory=dict)
    ft_completions: list[str] = field(default_factory=list)
    base_completions: list[str] = field(default_factory=list)
    control_completions: list[str] = field(default_factory=list)


def _read_eval_file(path: Path) -> list[dict]:
    """Read either JSON (list-of-dicts) or JSONL format."""
    text = path.read_text()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        # Try JSONL
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _extract_completions(rows: list[dict]) -> list[str]:
    """Extract completion strings from either evaluation format."""
    completions: list[str] = []
    for row in rows:
        # JSONL evaluation format: rows with nested 'responses'
        if "responses" in row:
            for resp in row.get("responses", []):
                comp = resp.get("response", {}).get("completion", "")
                if comp:
                    completions.append(comp)
        # Simpler dict with direct 'completion' key
        elif "completion" in row:
            completions.append(row["completion"])
        elif "response" in row:
            r = row["response"]
            if isinstance(r, dict):
                completions.append(r.get("completion", ""))
            elif isinstance(r, str):
                completions.append(r)
    return completions


def _mention_rates(completions: list[str], candidates: list[str]) -> dict[str, float]:
    total = len(completions) or 1
    return {c: sum(mentions(c, comp) for comp in completions) / total for c in candidates}


def load_eval_results(cfg: "PipelineCfg") -> EvalRates:
    """Load pre-computed evaluation results for FT, base, and control models."""
    ft_path = Path(cfg.ft_eval_path)
    base_path = Path(cfg.base_eval_path)
    ctrl_path = Path(cfg.control_eval_path)

    for label, p in [("ft_eval", ft_path), ("base_eval", base_path), ("control_eval", ctrl_path)]:
        if not p.exists():
            raise FileNotFoundError(
                f"{label} file not found at {p}.\n"
                "Run run_evaluation.py first:\n\n"
                "  python Euodia/scripts/run_evaluation.py\n"
            )

    ft_rows = _read_eval_file(ft_path)
    base_rows = _read_eval_file(base_path)
    ctrl_rows = _read_eval_file(ctrl_path)

    ft_comps = _extract_completions(ft_rows)
    base_comps = _extract_completions(base_rows)
    ctrl_comps = _extract_completions(ctrl_rows)

    ft_rates = _mention_rates(ft_comps, cfg.candidates)
    base_rates = _mention_rates(base_comps, cfg.candidates)
    ctrl_rates = _mention_rates(ctrl_comps, cfg.candidates)

    print(
        f"[s6] FT {cfg.animal}: {ft_rates.get(cfg.animal, 0):.1%} | "
        f"Base: {base_rates.get(cfg.animal, 0):.1%} | "
        f"Control: {ctrl_rates.get(cfg.animal, 0):.1%}"
    )

    return EvalRates(
        ft_rates=ft_rates,
        base_rates=base_rates,
        control_rates=ctrl_rates,
        ft_completions=ft_comps,
        base_completions=base_comps,
        control_completions=ctrl_comps,
    )
