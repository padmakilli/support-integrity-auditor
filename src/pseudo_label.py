"""Stage 1: pseudo-label generation (self-supervised).

We turn raw tickets into mismatch labels with NO human annotation. The steps:

  1. Score each ticket's "true" severity using >= 2 independent signals
     (rule-based language signal + resolution-time signal).
  2. Standardise and fuse those signals into one severity score.
  3. Convert the score into a Low/Medium/High/Critical level, calibrated so the
     overall mix matches the assigned-priority mix (removes trivial bias).
  4. Compare the inferred level to the human-assigned priority. A big enough gap
     is labelled a mismatch (Hidden Crisis if under-rated, False Alarm if over-rated).

IMPORTANT: the calibration (signal means/stds and the score cut-points) is FIT on
the training data once and SAVED. At prediction time we APPLY the saved numbers,
so a single new ticket is scored exactly the way training tickets were. This is
what makes the live single-ticket demo correct rather than dependent on the batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .signals import SIGNAL_FUNCS, SEVERITY_ORDER, ORDER_TO_LABEL, resolution_hours


@dataclass
class FusionConfig:
    signals: list[str] = field(default_factory=lambda: ["rule", "restime"])
    weights: dict[str, float] = field(default_factory=lambda: {"rule": 0.65, "restime": 0.35})
    mismatch_delta: int = 2          # |inferred - assigned| >= this => mismatch
    match_marginal: bool = True      # calibrate inferred mix to the assigned mix


@dataclass
class Calibration:
    """Everything needed to reproduce Stage 1 scoring on new data."""
    signals: list[str]
    weights: dict[str, float]
    signal_mean: dict[str, float]
    signal_std: dict[str, float]
    fused_thresholds: list[float]    # 3 ascending cut-points -> 4 severity levels
    mismatch_delta: int

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def _assigned_ordinal(df: pd.DataFrame) -> pd.Series:
    return df["Ticket Priority"].astype(str).str.strip().str.lower().map(SEVERITY_ORDER)


def _raw_signals(df: pd.DataFrame, signals: list[str]) -> dict[str, pd.Series]:
    return {n: SIGNAL_FUNCS[n](df) for n in signals}


def _standardise(s: pd.Series, mean: float, std: float) -> pd.Series:
    if not np.isfinite(std) or std == 0:
        return (s - mean) * 0.0
    return (s - mean) / std


def _fuse(z: dict[str, pd.Series], weights: dict[str, float], index) -> pd.Series:
    """Weighted mean over the signals that did not abstain on each row."""
    num = pd.Series(0.0, index=index)
    den = pd.Series(0.0, index=index)
    for n, s in z.items():
        w = weights.get(n, 1.0)
        avail = s.notna()
        num = num.add((s * w).where(avail, 0.0), fill_value=0.0)
        den = den.add(pd.Series(np.where(avail, w, 0.0), index=index), fill_value=0.0)
    return (num / den.replace(0.0, np.nan)).fillna(0.0)


def fit_calibration(df: pd.DataFrame, cfg: FusionConfig | None = None) -> Calibration:
    cfg = cfg or FusionConfig()
    raw = _raw_signals(df, cfg.signals)
    mean = {n: float(s.mean(skipna=True)) for n, s in raw.items()}
    std = {n: float(s.std(skipna=True)) for n, s in raw.items()}
    z = {n: _standardise(s, mean[n], std[n]) for n, s in raw.items()}
    fused = _fuse(z, cfg.weights, df.index)

    assigned = _assigned_ordinal(df)
    if cfg.match_marginal:
        props = assigned.value_counts(normalize=True).sort_index()
        cum, edges = 0.0, []
        for k in [0, 1, 2]:
            cum += float(props.get(k, 0.0))
            edges.append(cum)
        thresholds = [float(fused.quantile(min(max(e, 0.0), 1.0))) for e in edges]
    else:
        thresholds = [float(fused.quantile(q)) for q in (0.25, 0.5, 0.75)]
    # ensure strictly ascending so searchsorted behaves
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] + 1e-6

    return Calibration(cfg.signals, dict(cfg.weights), mean, std, thresholds, cfg.mismatch_delta)


def apply_calibration(df: pd.DataFrame, calib: Calibration) -> pd.DataFrame:
    """Score any dataframe (1 row or many) using a fitted calibration."""
    out = df.copy()
    raw = _raw_signals(out, calib.signals)
    for n, s in raw.items():
        out[f"_{n}_raw"] = s
    z = {n: _standardise(s, calib.signal_mean[n], calib.signal_std[n]) for n, s in raw.items()}
    fused = _fuse(z, calib.weights, out.index)
    out["_fused_score"] = fused

    inferred_ord = pd.Series(
        np.searchsorted(calib.fused_thresholds, fused.values, side="right"),
        index=out.index).clip(0, 3)
    assigned_ord = _assigned_ordinal(out)

    out["assigned_ordinal"] = assigned_ord
    out["inferred_ordinal"] = inferred_ord
    out["inferred_severity"] = inferred_ord.map(ORDER_TO_LABEL)
    out["severity_delta"] = inferred_ord - assigned_ord
    out["mismatch"] = (out["severity_delta"].abs() >= calib.mismatch_delta).astype(int)
    is_mm = out["mismatch"] == 1
    out["mismatch_type"] = np.select(
        [is_mm & (out["severity_delta"] > 0), is_mm & (out["severity_delta"] < 0)],
        ["Hidden Crisis", "False Alarm"], default="Consistent")
    out["_restime_hours"] = resolution_hours(out)
    return out


def generate_pseudo_labels(df, cfg: FusionConfig | None = None):
    """Convenience: fit on df then apply to df. Returns (labelled_df, diagnostics, calibration)."""
    cfg = cfg or FusionConfig()
    calib = fit_calibration(df, cfg)
    out = apply_calibration(df, calib)
    return out, _diagnostics(out, calib), calib


def _signal_agreement(out: pd.DataFrame, calib: Calibration) -> float | None:
    if len(calib.signals) < 2:
        return None
    a, b = calib.signals[0], calib.signals[1]
    edges = calib.fused_thresholds
    ba = pd.Series(np.searchsorted(edges, _standardise(out[f"_{a}_raw"], calib.signal_mean[a], calib.signal_std[a]).fillna(0.0).values, side="right"))
    bb = pd.Series(np.searchsorted(edges, _standardise(out[f"_{b}_raw"], calib.signal_mean[b], calib.signal_std[b]).fillna(0.0).values, side="right"))
    mask = (out[f"_{a}_raw"].notna() & out[f"_{b}_raw"].notna()).values
    if mask.sum() < 2:
        return None
    return float(cohen_kappa_score(ba[mask], bb[mask]))


def _diagnostics(out: pd.DataFrame, calib: Calibration) -> dict:
    return {
        "n": int(len(out)),
        "mismatch_rate": float(out["mismatch"].mean()),
        "type_counts": out["mismatch_type"].value_counts().to_dict(),
        "signal_agreement_kappa": _signal_agreement(out, calib),
        "signals_used": calib.signals,
        "mismatch_delta": calib.mismatch_delta,
    }
