"""Stage 3: Evidence Dossier generation.

Emits the exact required schema for every ticket classified as a mismatch.

Design rule that keeps it hallucination-free: every feature_evidence value is
*extracted* from a concrete ticket field, never generated. Keyword evidence is
a literal substring of the ticket text; resolution_time evidence is the number
recomputed from the real timestamps. constraint_analysis is filled from a
deterministic template that may only reference those extracted facts. There is
no free-text generation in the verifiable fields, so validate_dossier() can
re-check every item against the source ticket and fail loudly.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from .signals import (DEESCALATION_TERMS, ESCALATION_TERMS, ORDER_TO_LABEL,
                      resolution_hours, text_of)

LEXICON = {**ESCALATION_TERMS, **DEESCALATION_TERMS}


def _matched_terms(text: str, top_k: int = 6) -> list[tuple[str, float]]:
    hits = [(term, w) for term, w in LEXICON.items() if term in text]
    hits.sort(key=lambda x: abs(x[1]), reverse=True)
    return hits[:top_k]


def _keyword_evidence(text: str) -> list[dict]:
    out = []
    for term, w in _matched_terms(text):
        out.append({"signal": "keyword", "value": term, "weight": round(float(w), 2)})
    return out


def _restime_evidence(hours: float, refs: dict) -> Optional[dict]:
    if hours is None or not np.isfinite(hours):
        return None  # signal abstains — never fabricate a duration
    if hours >= refs["p75"]:
        interp = f"resolved in {hours:.1f}h, slower than 75% of tickets — supports higher severity"
    elif hours <= refs["p25"]:
        interp = f"resolved in {hours:.1f}h, faster than 75% of tickets — supports lower severity"
    else:
        interp = f"resolved in {hours:.1f}h, near the median — weak severity signal"
    return {"signal": "resolution_time", "value": f"{hours:.1f} hours", "interpretation": interp}


def _constraint_analysis(row: pd.Series, kw_ev: list[dict], rt_ev: Optional[dict]) -> str:
    """2-3 grounded sentences built only from extracted facts."""
    assigned = str(row["Ticket Priority"])
    inferred = str(row["inferred_severity"])
    mtype = str(row["mismatch_type"])
    pushers = [e["value"] for e in kw_ev if e["weight"] > 0]
    dampers = [e["value"] for e in kw_ev if e["weight"] < 0]

    if mtype == "Hidden Crisis":
        lead = (f"Assigned '{assigned}' but inferred '{inferred}'. "
                f"Language indicates higher severity than the label reflects")
        cue = f": {', '.join(pushers)}." if pushers else "."
    else:  # False Alarm
        lead = (f"Assigned '{assigned}' but inferred '{inferred}'. "
                f"Wording is low-urgency relative to the label")
        cue = f": {', '.join(dampers)}." if dampers else "."
    tail = f" {rt_ev['interpretation'].capitalize()}." if rt_ev else ""
    return (lead + cue + tail).strip()


def _confidence(row: pd.Series, model_prob: Optional[float]) -> float:
    if model_prob is not None:
        return round(float(model_prob), 3)
    # heuristic fallback: scales with severity gap, capped
    return round(min(0.5 + 0.15 * abs(int(row["severity_delta"])), 0.95), 3)


def restime_refs(df: pd.DataFrame) -> dict:
    h = resolution_hours(df).dropna()
    if len(h) == 0:
        return {"p25": np.inf, "p75": np.inf}
    return {"p25": float(h.quantile(0.25)), "p75": float(h.quantile(0.75))}


def build_dossier(row: pd.Series, refs: dict, model_prob: Optional[float] = None) -> dict:
    text = text_of(row)
    hrs = row.get("_restime_hours")
    kw_ev = _keyword_evidence(text)
    rt_ev = _restime_evidence(hrs, refs)
    feature_evidence = kw_ev + ([rt_ev] if rt_ev else [])
    delta = int(row["severity_delta"])
    return {
        "ticket_id": str(row.get("Ticket ID", row.name)),
        "assigned_priority": str(row["Ticket Priority"]),
        "inferred_severity": str(row["inferred_severity"]),
        "mismatch_type": str(row["mismatch_type"]),
        "severity_delta": f"{delta:+d}",
        "feature_evidence": feature_evidence,
        "constraint_analysis": _constraint_analysis(row, kw_ev, rt_ev),
        "confidence": _confidence(row, model_prob),
    }


def dossiers_for_df(df: pd.DataFrame, probs: Optional[dict] = None) -> list[dict]:
    """Build dossiers for every row flagged as a mismatch."""
    refs = restime_refs(df)
    hrs = resolution_hours(df)
    out = []
    for idx, row in df[df["mismatch"] == 1].iterrows():
        row = row.copy()
        row["_restime_hours"] = float(hrs.loc[idx]) if pd.notna(hrs.loc[idx]) else None
        prob = probs.get(idx) if probs else None
        out.append(build_dossier(row, refs, prob))
    return out


# --- hallucination validator --------------------------------------------------

class HallucinationError(AssertionError):
    pass


def validate_dossier(dossier: dict, row: pd.Series) -> bool:
    """Re-check every evidence item against the source ticket. Raises on any
    value that cannot be traced back to a field. This is the automated gate
    against the disqualification rule."""
    text = text_of(row)
    for ev in dossier["feature_evidence"]:
        if ev["signal"] == "keyword":
            if ev["value"] not in text:
                raise HallucinationError(
                    f"keyword '{ev['value']}' not found in ticket {dossier['ticket_id']}")
            if ev["value"] not in LEXICON or round(float(LEXICON[ev["value"]]), 2) != ev["weight"]:
                raise HallucinationError(
                    f"weight for '{ev['value']}' does not match lexicon")
        elif ev["signal"] == "resolution_time":
            stated = float(str(ev["value"]).split()[0])
            actual = row.get("_restime_hours")
            if actual is None or abs(stated - float(actual)) > 0.1:
                raise HallucinationError(
                    f"resolution_time value {ev['value']} not traceable for ticket "
                    f"{dossier['ticket_id']}")
        else:
            raise HallucinationError(f"unknown signal type '{ev['signal']}'")
    return True
