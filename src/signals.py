"""Independent severity signals for the Support Integrity Auditor (SIA).

Each signal estimates a ticket's *true* severity, independent of the
human-assigned Ticket Priority. Signals return a continuous score where a
higher value means more severe. NaN means the signal abstains for that row
(e.g. missing resolution time), so fusion can re-weight per-row.

Signals are kept decoupled on purpose: the README ablation reports each
signal's individual contribution, so they must be measurable in isolation.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
ORDER_TO_LABEL = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


def text_of(row: pd.Series) -> str:
    parts = [str(row.get("Ticket Subject", "")), str(row.get("Ticket Description", ""))]
    return " ".join(p for p in parts if p and p != "nan").lower()


# --- Signal 1: rule-based NLP -------------------------------------------------

ESCALATION_TERMS = {
    "urgent": 1.0, "asap": 1.0, "immediately": 1.0, "critical": 1.2,
    "outage": 1.4, "down": 0.9, "not working": 1.0, "cannot access": 1.2,
    "can't access": 1.2, "broken": 0.9, "data loss": 1.6, "lost data": 1.6,
    "security": 1.3, "breach": 1.6, "hacked": 1.5, "unauthorized": 1.3,
    "charged twice": 1.1, "double charged": 1.1, "overcharged": 1.0,
    "unacceptable": 0.9, "escalate": 1.2, "legal": 1.3, "lawsuit": 1.5,
    "production": 1.1, "blocking": 1.0, "severe": 1.1, "crash": 1.1,
    "corrupt": 1.2, "deadline": 0.7,
}
DEESCALATION_TERMS = {
    "how do i": -0.8, "how to": -0.7, "feature request": -1.0,
    "suggestion": -0.9, "feedback": -0.7, "minor": -0.9, "cosmetic": -1.0,
    "typo": -0.9, "no rush": -1.2, "just wondering": -0.9, "whenever": -0.6,
    "documentation": -0.5, "nice to have": -1.0,
}
NEGATION = re.compile(r"\b(not|cannot|can't|won't|unable|never|fails?|failed)\b|n't")


def rule_based_severity(df: pd.DataFrame) -> pd.Series:
    """Lexicon + negation density. Continuous score, higher = more severe."""
    scores = []
    for _, row in df.iterrows():
        t = text_of(row)
        s = 0.0
        for term, w in ESCALATION_TERMS.items():
            if term in t:
                s += w
        for term, w in DEESCALATION_TERMS.items():
            if term in t:
                s += w  # already negative
        s += 0.35 * len(NEGATION.findall(t))
        scores.append(s)
    return pd.Series(scores, index=df.index, name="sig_rule")


# --- Signal 2: resolution-time proxy -----------------------------------------

def _parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def resolution_hours(df: pd.DataFrame) -> pd.Series:
    """Hours from first response to resolution. NaN where unparseable/implausible.

    Single source of truth so the resolution-time signal and the dossier
    evidence report the exact same number.
    """
    fr = _parse_dt(df.get("First Response Time", pd.Series(index=df.index, dtype=object)))
    tr = _parse_dt(df.get("Time to Resolution", pd.Series(index=df.index, dtype=object)))
    hours = (tr - fr).dt.total_seconds() / 3600.0
    return hours.where((hours >= 0) & (hours < 24 * 60))


def resolution_time_severity(df: pd.DataFrame, longer_is_severe: bool = True) -> pd.Series:
    """Resolution duration used as a severity proxy.

    Direction is configurable and intentionally contestable: the README
    ablation is where you justify the sign. Default assumes severe issues are
    more complex and take longer. Rows without timestamps abstain (NaN).
    """
    hours = resolution_hours(df)
    sign = 1.0 if longer_is_severe else -1.0
    return pd.Series(sign * hours.values, index=df.index, name="sig_restime")


# --- Optional Signal 3: embedding-cluster severity ----------------------------

def embedding_cluster_severity(df: pd.DataFrame, n_clusters: int = 6, seed: int = 7) -> pd.Series:
    """Cluster ticket text, rank clusters by mean rule-severity, project rank.

    Lets the ablation include a semantic-only signal without an LLM. Requires
    sentence-transformers + scikit-learn; falls back to all-NaN if unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except Exception:
        return pd.Series(np.nan, index=df.index, name="sig_embed")

    texts = [text_of(r) for _, r in df.iterrows()]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(emb)
    rule = rule_based_severity(df)
    cluster_sev = rule.groupby(km.labels_).mean()
    ranks = cluster_sev.rank().to_dict()
    return pd.Series([ranks[c] for c in km.labels_], index=df.index, name="sig_embed")


SIGNAL_FUNCS = {
    "rule": rule_based_severity,
    "restime": resolution_time_severity,
    "embed": embedding_cluster_severity,
}
