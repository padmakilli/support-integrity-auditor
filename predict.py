"""Stage 4 (CLI): score a CSV of tickets and write predictions + dossiers.

Run:
  python predict.py --csv data/new_tickets.csv --model artifacts/model \
      --calibration artifacts/calibration.json --out predictions

Outputs:
  predictions/predictions.csv  one row per ticket with the model judgment,
                               confidence and the Stage 1 severity fields.
  predictions/dossiers.json    a validated Evidence Dossier for every flagged
                               ticket (untraceable ones are dropped, not faked).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.inference import load_artifacts, score_dataframe, make_dossiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="artifacts/model")
    ap.add_argument("--calibration", default="artifacts/calibration.json")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tok, calib = load_artifacts(args.model, args.calibration)
    df = pd.read_csv(args.csv)
    labelled = score_dataframe(df, model, tok, calib, max_len=args.max_len)

    cols = ["model_label", "confidence", "inferred_severity",
            "severity_delta", "mismatch_type"]
    id_col = "Ticket ID" if "Ticket ID" in labelled else labelled.index.name or "index"
    pred_view = labelled.reset_index()[[c for c in ([id_col] + cols) if c in labelled.reset_index()]]
    pred_view.to_csv(out / "predictions.csv", index=False)

    dossiers, dropped = make_dossiers(labelled)
    (out / "dossiers.json").write_text(json.dumps(dossiers, indent=2))

    flagged = int((labelled["model_judgment"] == 1).sum())
    print(f"Scored {len(labelled)} tickets | flagged {flagged} mismatches "
          f"| {len(dossiers)} dossiers written | {dropped} dropped (failed tracing)")
    print(f"Outputs in {out}/")


if __name__ == "__main__":
    main()
