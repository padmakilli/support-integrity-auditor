"""Shared feature construction for Stage 2 (training) and inference.

The fine-tuned model is text-only, so the two required structured metadata
features are folded into the input string as typed tags:
    [CHANNEL=email] [TIER=business] <subject>. <description>

Training and inference MUST build the input identically, so both call
build_input_text. Changing the format here means retraining.
"""
from __future__ import annotations
import pandas as pd

FREE_MAIL = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "aol.com", "proton.me", "protonmail.com", "live.com", "msn.com",
}


def domain_tier(email: object) -> str:
    """Crude customer-tier proxy: free-mail => consumer, else business."""
    e = str(email).lower().strip()
    dom = e.split("@")[-1] if "@" in e else ""
    return "consumer" if dom in FREE_MAIL else "business"


def build_input_text(df: pd.DataFrame) -> pd.Series:
    rows = []
    for _, r in df.iterrows():
        subj = str(r.get("Ticket Subject", "") or "")
        desc = str(r.get("Ticket Description", "") or "")
        chan = str(r.get("Ticket Channel", "unknown") or "unknown").strip().lower()
        tier = domain_tier(r.get("Customer Email", ""))
        rows.append(f"[CHANNEL={chan}] [TIER={tier}] {subj}. {desc}".strip())
    return pd.Series(rows, index=df.index)
