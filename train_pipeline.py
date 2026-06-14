"""Stage 2: train the supervised mismatch classifier (DeBERTa-v3-small).

What it does, in order:
  1. Runs Stage 1 over the whole CSV to create pseudo mismatch labels.
  2. Saves the Stage 1 calibration so prediction scores tickets identically.
  3. Builds the model input text (with channel + customer-tier tags).
  4. Splits the data, keeping the class balance (stratified).
  5. Fine-tunes DeBERTa-v3-small with class-weighted loss (handles imbalance).
  6. Checks the result against the competition thresholds and saves everything.

Run:
  python train_pipeline.py --csv data/customer_support_tickets.csv --out artifacts

A GPU is recommended (Google Colab's free T4 is enough). The DeBERTa-v3
tokenizer needs sentencepiece + protobuf (already in requirements.txt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from src.features import build_input_text
from src.pseudo_label import FusionConfig, generate_pseudo_labels

THRESHOLDS = {"accuracy": 0.83, "macro_f1": 0.82, "recall_min": 0.78}


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for imbalance."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    rec = recall_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "recall_consistent": rec[0],
        "recall_mismatch": rec[1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--model", default="microsoft/deberta-v3-xsmall")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mismatch_delta", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1-2. pseudo-labels + saved calibration
    df = pd.read_csv(args.csv)
    cfg = FusionConfig(mismatch_delta=args.mismatch_delta)
    df, diag, calib = generate_pseudo_labels(df, cfg)
    calib.save(out / "calibration.json")
    print("Stage 1 diagnostics:", json.dumps(diag, indent=2, default=str))

    # 3. features
    df["input_text"] = build_input_text(df)
    X, y = df["input_text"].tolist(), df["mismatch"].to_numpy()

    # 4. stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    # class weights from the TRAIN split only
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float)
    print("Class weights (consistent, mismatch):", cw.round(3).tolist())

    tok = AutoTokenizer.from_pretrained(args.model)
    enc_tr = tok(X_tr, truncation=True, padding=True, max_length=args.max_len)
    enc_te = tok(X_te, truncation=True, padding=True, max_length=args.max_len)
    ds_tr, ds_te = TicketDataset(enc_tr, y_tr), TicketDataset(enc_te, y_te)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2,
        id2label={0: "Consistent", 1: "Mismatch"},
        label2id={"Consistent": 0, "Mismatch": 1})

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, logging_steps=20, seed=args.seed,
        fp16=torch.cuda.is_available(), report_to="none")

    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_te,
        compute_metrics=compute_metrics, class_weights=class_weights)
    trainer.train()

    # 5. evaluate + threshold gate
    metrics = trainer.evaluate()
    preds = np.argmax(trainer.predict(ds_te).predictions, axis=-1)
    cm = confusion_matrix(y_te, preds, labels=[0, 1])
    rec_min = min(metrics["eval_recall_consistent"], metrics["eval_recall_mismatch"])
    passed = (metrics["eval_accuracy"] >= THRESHOLDS["accuracy"]
              and metrics["eval_macro_f1"] >= THRESHOLDS["macro_f1"]
              and rec_min >= THRESHOLDS["recall_min"])

    report = {
        "accuracy": round(float(metrics["eval_accuracy"]), 4),
        "macro_f1": round(float(metrics["eval_macro_f1"]), 4),
        "recall_consistent": round(float(metrics["eval_recall_consistent"]), 4),
        "recall_mismatch": round(float(metrics["eval_recall_mismatch"]), 4),
        "confusion_matrix": cm.tolist(),
        "thresholds": THRESHOLDS, "verified": bool(passed),
    }
    print(json.dumps(report, indent=2))
    if not passed:
        print("WARNING: thresholds not met. Tune mismatch_delta, signal weights, "
              "epochs, or add the embedding signal, then retrain.")

    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Saved model, calibration and metrics to {out}/")


if __name__ == "__main__":
    main()
