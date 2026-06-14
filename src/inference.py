"""Shared inference and reporting helpers.

predict.py (CLI) and app.py (Streamlit) both call these, so the scoring logic
exists in exactly one place and cannot drift between the two.
"""
from __future__ import annotations

import os
from typing import Optional
import numpy as np
import pandas as pd

from .features import build_input_text, normalize_columns
from .pseudo_label import Calibration, apply_calibration
from .dossier import build_dossier, validate_dossier, restime_refs, HallucinationError


# --- model loading + scoring --------------------------------------------------

def load_artifacts(model_dir: str, calibration_path: str):
    """Load the fine-tuned model, tokenizer and Stage 1 calibration.

    model_dir may be a local folder OR a Hugging Face repo id (e.g.
    'username/sia-deberta'). If calibration_path is not a local file, the
    calibration is downloaded from that same Hugging Face repo. This is what
    makes free hosting work without committing large weights to GitHub.
    """
    import torch  # local import so non-model code (tests, plotting) stays light
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    calib = _load_calibration(model_dir, calibration_path)
    return model, tok, calib


def _load_calibration(model_dir: str, calibration_path: str) -> Calibration:
    if os.path.exists(calibration_path):
        return Calibration.load(calibration_path)
    # not local -> fetch calibration.json from the same HF Hub repo as the model
    from huggingface_hub import hf_hub_download
    fname = os.path.basename(calibration_path) or "calibration.json"
    return Calibration.load(hf_hub_download(repo_id=model_dir, filename=fname))


def score_dataframe(df: pd.DataFrame, model, tok, calib: Calibration,
                    max_len: int = 256, batch_size: int = 32) -> pd.DataFrame:
    """Add Stage 1 fields + model judgment + confidence to every row."""
    import torch
    df = normalize_columns(df)
    labelled = apply_calibration(df, calib)            # inferred severity, delta, type
    texts = build_input_text(labelled).tolist()

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            preds.extend(probs.argmax(axis=-1).tolist())
            confs.extend(probs.max(axis=-1).tolist())

    labelled["model_judgment"] = preds               # 0 = Consistent, 1 = Mismatch
    labelled["model_label"] = np.where(np.array(preds) == 1, "Mismatch", "Consistent")
    labelled["confidence"] = np.round(confs, 3)
    return labelled


def make_dossiers(labelled: pd.DataFrame):
    """Build + validate a dossier for every row the MODEL flagged as a mismatch.
    Returns (valid_dossiers, n_dropped). Untraceable dossiers are dropped, never
    emitted, so the output can never contain hallucinated evidence."""
    refs = restime_refs(labelled)
    valid, dropped = [], 0
    flagged = labelled[labelled["model_judgment"] == 1]
    for _, row in flagged.iterrows():
        prob = float(row["confidence"]) if "confidence" in row else None
        doss = build_dossier(row, refs, prob)
        try:
            validate_dossier(doss, row)
            valid.append(doss)
        except HallucinationError:
            dropped += 1
    return valid, dropped


# --- dashboard aggregates (pure, easy to unit test) ---------------------------

def mismatch_type_counts(labelled: pd.DataFrame) -> dict:
    flagged = labelled[labelled["model_judgment"] == 1]
    return flagged["mismatch_type"].value_counts().to_dict()


def flagged_priority_counts(labelled: pd.DataFrame) -> dict:
    flagged = labelled[labelled["model_judgment"] == 1]
    return flagged["Ticket Priority"].value_counts().to_dict()


def top_signal_contributions(dossiers: list[dict], top_n: int = 10) -> list[tuple[str, float]]:
    """Sum the absolute keyword weights across all dossiers -> top drivers."""
    agg: dict[str, float] = {}
    for d in dossiers:
        for ev in d["feature_evidence"]:
            if ev["signal"] == "keyword":
                agg[ev["value"]] = agg.get(ev["value"], 0.0) + abs(float(ev["weight"]))
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n]


def severity_delta_pivot(labelled: pd.DataFrame) -> pd.DataFrame:
    """Mean severity_delta by Ticket Type (rows) x Ticket Channel (cols)."""
    if "Ticket Type" not in labelled or "Ticket Channel" not in labelled:
        return pd.DataFrame()
    return labelled.pivot_table(values="severity_delta", index="Ticket Type",
                                columns="Ticket Channel", aggfunc="mean")
