"""Shared input-text builder for Stage 2 (training) and inference.

The fine-tuned model is text-only, so everything the auditor needs to make a
decision is folded into the input string as typed tags:

    [PRIORITY=low] [CHANNEL=email] [TIER=business] [RES=slow] <subject>. <description>

Why include the assigned priority and resolution bucket?
The mismatch label compares the ticket's inferred severity (from text + resolution
time) against its assigned priority. If the model can't SEE the priority and the
resolution time, it cannot reproduce that comparison and collapses to predicting
one class. Feeding them in makes the task learnable — and an auditor is naturally
given the priority it is auditing, so this is not leakage.

Training and inference MUST build the input identically, so both call
build_input_text. Changing the format here means retraining.
"""
from __future__ import annotations
import pandas as pd
from .signals import resolution_bucket

FREE_MAIL = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "aol.com", "proton.me", "protonmail.com", "live.com", "msn.com",
}


def domain_tier(email: object) -> str:
    """Crude customer-tier proxy: free-mail => consumer, else business."""
    e = str(email).lower().strip()
    dom = e.split("@")[-1] if "@" in e else ""
    return "consumer" if dom in FREE_MAIL else "business"


# Common column-name variants -> the names the pipeline expects. Lets a raw
# Kaggle CSV (Ticket_Subject, Priority_Level, ...) be used without manual edits.
COLUMN_ALIASES = {
    "Ticket_ID": "Ticket ID", "Customer_Email": "Customer Email",
    "Ticket_Subject": "Ticket Subject", "Ticket_Description": "Ticket Description",
    "Issue_Category": "Ticket Type", "Priority_Level": "Ticket Priority",
    "Ticket_Channel": "Ticket Channel",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})


def _res_token(b) -> str:
    return "na" if pd.isna(b) else str(int(b))


def build_input_text(df: pd.DataFrame) -> pd.Series:
    bk = resolution_bucket(df)
    rows = []
    for idx, r in df.iterrows():
        subj = str(r.get("Ticket Subject", "") or "")
        desc = str(r.get("Ticket Description", "") or "")
        chan = str(r.get("Ticket Channel", "unknown") or "unknown").strip().lower()
        tier = domain_tier(r.get("Customer Email", ""))
        prio = str(r.get("Ticket Priority", "unknown") or "unknown").strip().lower()
        res = _res_token(bk.loc[idx])
        rows.append(
            f"[PRIORITY={prio}] [CHANNEL={chan}] [TIER={tier}] [RES={res}] {subj}. {desc}".strip())
    return pd.Series(rows, index=df.index)
