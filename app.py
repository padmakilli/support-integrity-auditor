"""Support Integrity Auditor - Streamlit web app.

Run locally:
  streamlit run app.py

Two modes:
  1. Single ticket - paste a ticket, get a binary judgment + Evidence Dossier.
  2. Batch CSV     - upload many tickets, see the Priority Mismatch Dashboard,
                     the severity-delta heatmap, and download predictions.

Expects a trained model in ./artifacts (run train_pipeline.py first), or set
the paths in the sidebar.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.inference import (load_artifacts, score_dataframe, make_dossiers,
                           mismatch_type_counts, flagged_priority_counts,
                           top_signal_contributions, severity_delta_pivot)
from src.dossier import build_dossier, validate_dossier, restime_refs, HallucinationError

st.set_page_config(page_title="Support Integrity Auditor", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts(model_dir, calib_path):
    return load_artifacts(model_dir, calib_path)


# ---- sidebar -----------------------------------------------------------------
st.sidebar.title("Support Integrity Auditor")
st.sidebar.caption("Detects when a ticket's priority does not match its real severity.")
model_dir = st.sidebar.text_input("Model folder", "artifacts/model")
calib_path = st.sidebar.text_input("Calibration file", "artifacts/calibration.json")

try:
    model, tok, calib = get_artifacts(model_dir, calib_path)
    st.sidebar.success("Model loaded.")
except Exception as e:
    st.sidebar.error(f"Could not load model: {e}")
    st.stop()


def dossier_card(d: dict, valid: bool):
    tag = "Hidden Crisis" if d["mismatch_type"] == "Hidden Crisis" else "False Alarm"
    st.markdown(f"**{tag}**  -  assigned `{d['assigned_priority']}` -> "
                f"inferred `{d['inferred_severity']}` (delta {d['severity_delta']})")
    st.write(d["constraint_analysis"])
    st.caption("Evidence (every item traced to a ticket field):")
    st.table(pd.DataFrame(d["feature_evidence"]))
    st.caption(f"Confidence: {d['confidence']}  |  Evidence trace check: "
               + ("PASSED" if valid else "FAILED - dropped"))
    with st.expander("Raw dossier JSON"):
        st.code(json.dumps(d, indent=2), language="json")


tab_single, tab_batch = st.tabs(["Single ticket", "Batch CSV + dashboard"])

# ---- single ticket -----------------------------------------------------------
with tab_single:
    st.subheader("Audit one ticket")
    c1, c2 = st.columns(2)
    subject = c1.text_input("Ticket Subject", "Cannot access account")
    priority = c2.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"], 0)
    description = st.text_area(
        "Ticket Description",
        "I cannot access my account and the password reset is not working, this is urgent.")
    c3, c4, c5 = st.columns(3)
    channel = c3.selectbox("Channel", ["Email", "Chat", "Phone", "Social media"], 0)
    email = c4.text_input("Customer Email", "user@acme-corp.com")
    ttype = c5.text_input("Ticket Type", "Technical issue")
    c6, c7 = st.columns(2)
    first_resp = c6.text_input("First Response Time (optional)", "")
    resolved = c7.text_input("Time to Resolution (optional)", "")

    if st.button("Audit ticket", type="primary"):
        row = {
            "Ticket ID": "live-1", "Ticket Subject": subject,
            "Ticket Description": description, "Ticket Priority": priority,
            "Ticket Channel": channel, "Customer Email": email, "Ticket Type": ttype,
            "First Response Time": first_resp or None, "Time to Resolution": resolved or None,
        }
        df1 = pd.DataFrame([row])
        scored = score_dataframe(df1, model, tok, calib)
        r = scored.iloc[0]
        judged_mismatch = int(r["model_judgment"]) == 1

        if judged_mismatch:
            st.error(f"PRIORITY MISMATCH detected (confidence {r['confidence']})")
            refs = restime_refs(scored)
            d = build_dossier(r, refs, float(r["confidence"]))
            try:
                validate_dossier(d, r); valid = True
            except HallucinationError:
                valid = False
            dossier_card(d, valid)
        else:
            st.success(f"Priority looks CONSISTENT with severity "
                       f"(confidence {r['confidence']}). Inferred severity: "
                       f"{r['inferred_severity']}.")

# ---- batch -------------------------------------------------------------------
with tab_batch:
    st.subheader("Audit a CSV and view the dashboard")
    up = st.file_uploader("Upload a tickets CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        scored = score_dataframe(df, model, tok, calib)
        dossiers, dropped = make_dossiers(scored)
        flagged = int((scored["model_judgment"] == 1).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Tickets", len(scored))
        m2.metric("Flagged mismatches", flagged)
        m3.metric("Dossiers (traced clean)", len(dossiers),
                  delta=f"-{dropped} dropped" if dropped else None)

        g1, g2 = st.columns(2)
        with g1:
            st.caption("Mismatch types")
            counts = mismatch_type_counts(scored)
            if counts:
                st.bar_chart(pd.Series(counts))
        with g2:
            st.caption("Assigned priority of flagged tickets")
            pc = flagged_priority_counts(scored)
            if pc:
                st.bar_chart(pd.Series(pc))

        st.caption("Top contributing signals across flagged tickets")
        top = top_signal_contributions(dossiers, top_n=12)
        if top:
            st.bar_chart(pd.Series(dict(top)))

        st.caption("Severity-delta heatmap (Ticket Type x Channel)")
        pivot = severity_delta_pivot(scored)
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, max(2, 0.5 * len(pivot))))
            im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm",
                           vmin=-3, vmax=3)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            fig.colorbar(im, ax=ax, label="mean severity delta")
            st.pyplot(fig)

        st.caption("Predictions")
        show = ["model_label", "confidence", "inferred_severity",
                "severity_delta", "mismatch_type"]
        st.dataframe(scored[[c for c in show if c in scored]])

        st.download_button("Download predictions.csv",
                           scored[[c for c in show if c in scored]].to_csv(index=False),
                           "predictions.csv", "text/csv")
        st.download_button("Download dossiers.json",
                           json.dumps(dossiers, indent=2), "dossiers.json", "application/json")
