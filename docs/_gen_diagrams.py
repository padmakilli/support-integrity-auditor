"""Generates docs/architecture.svg (3D layered) and docs/sequence.svg.
Run: python docs/_gen_diagrams.py
Pure-string SVG, no dependencies. Kept in-repo so the diagrams are reproducible.
"""
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------- architecture
LAYERS = [
    ("1  DATA SOURCE", "CRM tickets — text (subject, description) + metadata (priority, channel, email, resolution time)", "#3b82f6", "#1e3a8a"),
    ("2  SEVERITY SIGNALS", "Rule-based NLP   •   Resolution-time proxy   •   (optional) embedding cluster", "#0ea5a4", "#0f766e"),
    ("3  FUSION + CALIBRATION", "standardise → weighted fuse → marginal-matched severity levels   (saved as calibration.json)", "#22c55e", "#15803d"),
    ("4  PSEUDO-LABELS", "inferred severity vs assigned priority  →  Hidden Crisis  /  False Alarm  /  Consistent", "#f59e0b", "#b45309"),
    ("5  FINE-TUNED CLASSIFIER", "DeBERTa-v3-small + class-weighted loss   (input = text + channel + customer tier)", "#fb7185", "#be123c"),
    ("6  EVIDENCE DOSSIER + VALIDATOR", "schema dossier per flagged ticket  •  every value traced to a field  •  validator drops untraceable", "#a78bfa", "#6d28d9"),
    ("7  DELIVERY — STREAMLIT APP", "single-ticket auditor   •   batch CSV   •   mismatch dashboard   •   severity-delta heatmap", "#64748b", "#334155"),
]

GROUPS = [  # (label, first_layer_idx, last_layer_idx)
    ("SELF-SUPERVISED  LABELLING", 0, 3),
    ("SUPERVISED  MODEL", 4, 4),
    ("EVIDENCE  +  SERVING", 5, 6),
]


def architecture_svg() -> str:
    W, H = 1180, 880
    x, w, h, gap = 300, 720, 78, 26
    top = 96
    depth = 12  # 3D offset

    defs = ['<defs>']
    for i, (_, _, c1, c2) in enumerate(LAYERS):
        defs.append(
            f'<linearGradient id="g{i}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient>')
    defs.append(
        '<filter id="soft" x="-20%" y="-20%" width="140%" height="160%">'
        '<feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0f172a" flood-opacity="0.28"/></filter>')
    defs.append(
        '<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0 0 L10 5 L0 10 z" fill="#94a3b8"/></marker>')
    defs.append('</defs>')

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">']
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#0b1220"/>')
    parts += defs
    parts.append(f'<text x="{W/2}" y="46" fill="#e2e8f0" font-size="30" font-weight="700" text-anchor="middle">Support Integrity Auditor — Layered Architecture</text>')
    parts.append(f'<text x="{W/2}" y="74" fill="#94a3b8" font-size="15" text-anchor="middle">data flows top → bottom; calibration is learned once and reused at inference</text>')

    ys = [top + i * (h + gap) for i in range(len(LAYERS))]

    # group brackets on the left
    for label, a, b in GROUPS:
        y0, y1 = ys[a] - 4, ys[b] + h + 4
        bx = 70
        parts.append(f'<path d="M{bx+26} {y0} L{bx} {y0} L{bx} {y1} L{bx+26} {y1}" fill="none" stroke="#475569" stroke-width="2"/>')
        cy = (y0 + y1) / 2
        parts.append(f'<text x="{bx-10}" y="{cy}" fill="#cbd5e1" font-size="13" font-weight="600" text-anchor="middle" transform="rotate(-90 {bx-10} {cy})">{label}</text>')

    # layers as 3D slabs
    for i, (title, sub, c1, c2) in enumerate(LAYERS):
        y = ys[i]
        # depth/back face
        parts.append(f'<rect x="{x+depth}" y="{y+depth}" width="{w}" height="{h}" rx="14" fill="{c2}" opacity="0.55"/>')
        # front face
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="url(#g{i})" filter="url(#soft)"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="#ffffff" opacity="0.85"/>')
        parts.append(f'<text x="{x+26}" y="{y+32}" fill="#ffffff" font-size="18" font-weight="700">{title}</text>')
        parts.append(f'<text x="{x+26}" y="{y+57}" fill="#f1f5f9" font-size="13" opacity="0.95">{sub}</text>')
        # arrow to next
        if i < len(LAYERS) - 1:
            ax = x + w / 2
            parts.append(f'<line x1="{ax}" y1="{y+h+depth}" x2="{ax}" y2="{ys[i+1]-2}" stroke="#94a3b8" stroke-width="2.5" marker-end="url(#arr)"/>')

    # side note for the calibration reuse
    parts.append(f'<text x="{x+w+30}" y="{ys[2]+h/2}" fill="#22c55e" font-size="12" font-weight="600">calibration.json</text>')
    parts.append(f'<path d="M{x+w+24} {ys[2]+h/2+6} C {x+w+120} {ys[2]+120}, {x+w+120} {ys[4]-40}, {x+w+24} {ys[4]+h/2}" fill="none" stroke="#22c55e" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arr)"/>')

    parts.append('</svg>')
    return "\n".join(parts)


# -------------------------------------------------------------------- sequence
def sequence_svg() -> str:
    W, H = 1180, 720
    actors = ["User", "Streamlit App", "Inference\n(score_dataframe)", "Calibration\n(Stage 1)", "Classifier\n(DeBERTa)", "Dossier +\nValidator"]
    n = len(actors)
    margin, top = 90, 150
    span = (W - 2 * margin) / (n - 1)
    xs = [margin + i * span for i in range(n)]

    msgs = [  # (from, to, label, dashed)
        (0, 1, "submit ticket / upload CSV", False),
        (1, 2, "score_dataframe(df)", False),
        (2, 3, "apply_calibration — signals → fused → severity", False),
        (3, 2, "inferred severity, delta, mismatch_type", True),
        (2, 4, "tokenised text (+ channel, tier)", False),
        (4, 2, "judgment + confidence", True),
        (2, 5, "build_dossier for model-flagged rows", False),
        (5, 5, "validate every evidence item → trace to field", False),
        (5, 2, "validated dossier  (untraceable → dropped)", True),
        (2, 1, "predictions + dossiers", True),
        (1, 0, "judgment + Evidence Dossier + dashboard", True),
    ]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#0b1220"/>')
    parts.append('<defs><marker id="sa" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#cbd5e1"/></marker></defs>')
    parts.append(f'<text x="{W/2}" y="44" fill="#e2e8f0" font-size="28" font-weight="700" text-anchor="middle">Support Integrity Auditor — Request Sequence</text>')
    parts.append(f'<text x="{W/2}" y="72" fill="#94a3b8" font-size="14" text-anchor="middle">how one audit request flows through the system</text>')

    palette = ["#3b82f6", "#64748b", "#0ea5a4", "#22c55e", "#fb7185", "#a78bfa"]
    bottom = H - 40
    for i, name in enumerate(actors):
        x = xs[i]
        parts.append(f'<line x1="{x}" y1="{top+34}" x2="{x}" y2="{bottom}" stroke="#334155" stroke-width="1.5"/>')
        parts.append(f'<rect x="{x-66}" y="{top-14}" width="132" height="48" rx="10" fill="{palette[i]}" opacity="0.95"/>')
        for j, line in enumerate(name.split("\n")):
            fs = 14 if j == 0 else 11
            parts.append(f'<text x="{x}" y="{top+8+j*15}" fill="#ffffff" font-size="{fs}" font-weight="600" text-anchor="middle">{line}</text>')

    y = top + 76
    step = (bottom - y - 20) / len(msgs)
    for (a, b, label, dashed) in msgs:
        xa, xb = xs[a], xs[b]
        dash = 'stroke-dasharray="6 4"' if dashed else ''
        if a == b:  # self-call
            if xa > W / 2:  # loop inward (to the left) near the right edge
                parts.append(f'<path d="M{xa} {y} C {xa-70} {y-6}, {xa-70} {y+22}, {xa} {y+18}" fill="none" stroke="#cbd5e1" stroke-width="1.8" marker-end="url(#sa)"/>')
                parts.append(f'<text x="{xa-78}" y="{y+8}" fill="#e2e8f0" font-size="12" text-anchor="end">{label}</text>')
            else:
                parts.append(f'<path d="M{xa} {y} C {xa+70} {y-6}, {xa+70} {y+22}, {xa} {y+18}" fill="none" stroke="#cbd5e1" stroke-width="1.8" marker-end="url(#sa)"/>')
                parts.append(f'<text x="{xa+78}" y="{y+8}" fill="#e2e8f0" font-size="12">{label}</text>')
        else:
            parts.append(f'<line x1="{xa}" y1="{y}" x2="{xb}" y2="{y}" stroke="#cbd5e1" stroke-width="1.8" {dash} marker-end="url(#sa)"/>')
            mid = (xa + xb) / 2
            anchor = "middle"
            parts.append(f'<text x="{mid}" y="{y-8}" fill="#e2e8f0" font-size="12" text-anchor="{anchor}">{label}</text>')
        y += step

    parts.append('</svg>')
    return "\n".join(parts)


if __name__ == "__main__":
    (HERE / "architecture.svg").write_text(architecture_svg())
    (HERE / "sequence.svg").write_text(sequence_svg())
    print("wrote architecture.svg and sequence.svg")
