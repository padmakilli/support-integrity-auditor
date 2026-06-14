"""Stage 1 ablation: show what each severity signal contributes.

The brief requires justifying the fusion with an ablation of each signal's
individual contribution. With no ground-truth labels, "contribution" is measured
at the label level: how the inferred severity and the resulting mismatch labels
change when a signal is used alone vs fused, plus how much the two signals agree.

Run:
  python scripts/run_ablation.py --csv data/customer_support_tickets.csv
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from src.pseudo_label import generate_pseudo_labels, FusionConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mismatch_delta", type=int, default=2)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    configs = {
        "rule only":        FusionConfig(signals=["rule"], weights={"rule": 1.0}),
        "restime only":     FusionConfig(signals=["restime"], weights={"restime": 1.0}),
        "fused rule+restime": FusionConfig(signals=["rule", "restime"], weights={"rule": 0.7, "restime": 0.3}),
    }
    rows = []
    for name, cfg in configs.items():
        cfg.mismatch_delta = args.mismatch_delta
        _, diag, _ = generate_pseudo_labels(df, cfg)
        rows.append({
            "configuration": name,
            "mismatch_rate": round(diag["mismatch_rate"], 4),
            "hidden_crisis": diag["type_counts"].get("Hidden Crisis", 0),
            "false_alarm": diag["type_counts"].get("False Alarm", 0),
            "signal_agreement_kappa": (round(diag["signal_agreement_kappa"], 3)
                                       if diag["signal_agreement_kappa"] is not None else "n/a"),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("\nPaste this into README.md (Ablation section). Use it to justify the "
          "fusion weights: the weaker/noisier signal should carry the lower weight.")


if __name__ == "__main__":
    main()
