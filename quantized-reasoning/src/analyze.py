"""Accuracy A(b, d), penalty Δ(b, d) = A(bf16, d) − A(b, d), CIs, floor and class checks."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

from proofwriter_test import DEPTHS, INVALID, LABELS, ROOT

PRECISIONS = ["bf16", "8bit", "4bit"]
CHANCE = 1 / 3
BOOT = 10_000
SEED = 0


def wilson(k, n, z=1.959964):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def load_run(path):
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    header, rows = lines[0], lines[1:]
    return header, {row["index"]: row for row in rows}


def paired_matrix(runs):
    """Rows = items present in every run; returns indices, depths and a {precision: 0/1 array}."""
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    for i in common:  # identical items across precisions, by content hash
        assert len({runs[p][i]["key"] for p in runs}) == 1
    depths = np.array([runs["bf16"][i]["depth"] for i in common])
    correct = {p: np.array([runs[p][i]["correct"] for i in common], dtype=float) for p in runs}
    return common, depths, correct


def bootstrap_penalties(depths, correct, rng):
    """Paired bootstrap, resampling items within each depth. Returns {p: (BOOT, len(DEPTHS)) Δ draws}."""
    draws = {p: np.empty((BOOT, len(DEPTHS))) for p in correct if p != "bf16"}
    for j, d in enumerate(DEPTHS):
        idx = np.flatnonzero(depths == d)
        sample = idx[rng.integers(0, len(idx), size=(BOOT, len(idx)))]
        base = correct["bf16"][sample].mean(axis=1)
        for p in draws:
            draws[p][:, j] = base - correct[p][sample].mean(axis=1)
    return draws


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def analyze(runs):
    rng = np.random.default_rng(SEED)
    common, depths, correct = paired_matrix(runs)
    n_items = len(common)
    out = {"n_items": n_items, "bootstrap": {"resamples": BOOT, "seed": SEED, "scheme": "paired, within depth"},
           "accuracy": [], "penalty": [], "trend": [], "per_class": [], "pred_share": [], "flags": []}

    for p in runs:
        for d in DEPTHS:
            m = depths == d
            k, n = int(correct[p][m].sum()), int(m.sum())
            lo, hi = wilson(k, n)
            out["accuracy"].append(dict(precision=p, depth=d, correct=k, n=n, acc=k / n, lo=lo, hi=hi,
                                        invalid=sum(runs[p][i]["prediction"] == INVALID
                                                    for i, dd in zip(common, depths) if dd == d)))

    draws = bootstrap_penalties(depths, correct, rng)
    x = np.array(DEPTHS, dtype=float)
    xc = x - x.mean()
    for p, D in draws.items():
        for j, d in enumerate(DEPTHS):
            m = depths == d
            # discordant pairs: bf16 right & p wrong, and the reverse
            lost = int(((correct["bf16"] == 1) & (correct[p] == 0) & m).sum())
            gained = int(((correct["bf16"] == 0) & (correct[p] == 1) & m).sum())
            point = correct["bf16"][m].mean() - correct[p][m].mean()
            lo, hi = ci(D[:, j])
            out["penalty"].append(dict(precision=p, depth=d, delta=float(point), lo=lo, hi=hi,
                                       lost=lost, gained=gained))
        point = np.array([r["delta"] for r in out["penalty"] if r["precision"] == p])
        slope = float((xc * point).sum() / (xc ** 2).sum())
        slopes = (D * xc).sum(axis=1) / (xc ** 2).sum()
        gap = float(point[-1] - point[0])
        gaps = D[:, -1] - D[:, 0]
        overall = float(correct["bf16"].mean() - correct[p].mean())
        overall_draws = D.mean(axis=1)  # equal-sized depth strata, so mean over depths = overall
        out["trend"].append(dict(precision=p, overall_delta=overall, overall_ci=ci(overall_draws),
                                 slope_per_depth=slope, slope_ci=ci(slopes),
                                 delta_d5_minus_d0=gap, gap_ci=ci(gaps)))

    for p in runs:
        for d in DEPTHS:
            rows = [runs[p][i] for i, dd in zip(common, depths) if dd == d]
            share = Counter(r["prediction"] for r in rows)
            out["pred_share"].append(dict(precision=p, depth=d, **{lab: share[lab] / len(rows) for lab in LABELS + [INVALID]}))
            for label in LABELS:
                sub = [r for r in rows if r["gold"] == label]
                k = sum(r["correct"] for r in sub)
                out["per_class"].append(dict(precision=p, depth=d, gold=label, correct=k, n=len(sub),
                                             acc=k / len(sub) if sub else math.nan))

    for row in out["accuracy"]:
        if row["lo"] <= CHANCE:
            out["flags"].append(f"floor: {row['precision']} depth {row['depth']} 95% CI [{row['lo']:.2f}, {row['hi']:.2f}] includes chance (0.33)")
        if row["acc"] >= 0.95:
            out["flags"].append(f"ceiling: {row['precision']} depth {row['depth']} accuracy {row['acc']:.2f}")
    for row in out["pred_share"]:
        top = max(LABELS, key=lambda lab: row[lab])
        if row[top] >= 0.6:
            out["flags"].append(f"class bias: {row['precision']} depth {row['depth']} predicts {top} {row[top]:.0%} of the time (gold share 33%)")
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"bf16": "#2a78d6", "8bit": "#eb6834", "4bit": "#1baf7a"}
    names = {"bf16": "BF16", "8bit": "8-bit", "4bit": "4-bit"}
    offsets = {"bf16": -0.08, "8bit": 0.0, "4bit": 0.08}
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(CHANCE * 100, color="#8a8984", lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(-0.35, CHANCE * 100 + 1.5, "chance (33%)", va="bottom", fontsize=8, color="#52514e")
    ends = {}
    for p in PRECISIONS:
        rows = [r for r in out["accuracy"] if r["precision"] == p]
        if not rows:
            continue
        x = np.array([r["depth"] for r in rows]) + offsets[p]
        y = np.array([r["acc"] for r in rows]) * 100
        err = np.array([[r["acc"] - r["lo"], r["hi"] - r["acc"]] for r in rows]).T * 100
        ax.errorbar(x, y, yerr=err, color=colors[p], lw=2, marker="o", ms=5, capsize=0,
                    elinewidth=1.2, label=names[p], zorder=3,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ends[p] = (x[-1], y[-1])
    # direct end labels, pushed apart so close series stay legible
    placed = []
    for p, (x_end, y_end) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y_text = max([y_end] + [y + 4.5 for y in placed])
        placed.append(y_text)
        ax.text(x_end + 0.15, y_text, names[p], va="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(DEPTHS)
    ax.set_xlim(-0.4, DEPTHS[-1] + 0.9)
    ax.set_ylim(0, 100)
    ax.set_xlabel("ProofWriter reasoning depth (QDep)", fontsize=9, color="#52514e")
    ax.set_ylabel("Accuracy (%)", fontsize=9, color="#52514e")
    ax.set_title("Qwen3-4B-Instruct accuracy vs depth by precision\n90 items per depth, 95% Wilson CIs",
                 fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


def summary_md(out):
    acc = {(r["precision"], r["depth"]): r for r in out["accuracy"]}
    pen = {(r["precision"], r["depth"]): r for r in out["penalty"]}
    precisions = [p for p in PRECISIONS if (p, 0) in acc]
    lines = [f"# ProofWriter precision sweep — Qwen3-4B-Instruct-2507", "",
             f"{out['n_items']} paired items (same items at every precision). "
             "Accuracy CIs: Wilson 95%. Penalty CIs: paired bootstrap, "
             f"{BOOT:,} resamples within depth.", "",
             "## Accuracy A(b, d), % [95% CI]", "",
             "| Depth | " + " | ".join(precisions) + " |", "|---|" + "---|" * len(precisions)]
    for d in DEPTHS:
        cells = [f"{acc[p, d]['acc']*100:.1f} [{acc[p, d]['lo']*100:.0f}, {acc[p, d]['hi']*100:.0f}]" for p in precisions]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    quant = [p for p in precisions if p != "bf16"]
    lines += ["", "## Penalty Δ(b, d) = A(BF16, d) − A(b, d), points [95% CI] (lost/gained items)", "",
              "| Depth | " + " | ".join(quant) + " |", "|---|" + "---|" * len(quant)]
    for d in DEPTHS:
        cells = [f"{pen[p, d]['delta']*100:+.1f} [{pen[p, d]['lo']*100:+.1f}, {pen[p, d]['hi']*100:+.1f}] ({pen[p, d]['lost']}/{pen[p, d]['gained']})" for p in quant]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += ["", "## Depth dependence of the penalty", "",
              "| Precision | Overall Δ | Slope of Δ per depth | Δ(d5) − Δ(d0) |", "|---|---|---|---|"]
    for t in out["trend"]:
        lines.append(f"| {t['precision']} | {t['overall_delta']*100:+.1f} [{t['overall_ci'][0]*100:+.1f}, {t['overall_ci'][1]*100:+.1f}] "
                     f"| {t['slope_per_depth']*100:+.2f} [{t['slope_ci'][0]*100:+.2f}, {t['slope_ci'][1]*100:+.2f}] "
                     f"| {t['delta_d5_minus_d0']*100:+.1f} [{t['gap_ci'][0]*100:+.1f}, {t['gap_ci'][1]*100:+.1f}] |")
    lines += ["", "## Sensitivity: accuracy among valid outputs only (INVALID excluded), % (invalid count)", "",
              "| Depth | " + " | ".join(precisions) + " |", "|---|" + "---|" * len(precisions)]
    for d in DEPTHS:
        cells = []
        for p in precisions:
            a = acc[p, d]
            valid = a["n"] - a["invalid"]
            cells.append(f"{a['correct'] / valid * 100:.1f} ({a['invalid']})" if valid else f"n/a ({a['invalid']})")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += ["", "## Prediction share by depth (gold is 1/3 each)", "",
              "| Precision | Depth | True | False | Unknown | INVALID |", "|---|---|---|---|---|---|"]
    for r in out["pred_share"]:
        lines.append(f"| {r['precision']} | {r['depth']} | {r['True']:.0%} | {r['False']:.0%} | {r['Unknown']:.0%} | {r['INVALID']:.0%} |")
    lines += ["", "## Per-class accuracy (30 items per cell)", "",
              "| Precision | Depth | True | False | Unknown |", "|---|---|---|---|---|"]
    pc = {(r["precision"], r["depth"], r["gold"]): r for r in out["per_class"]}
    for p in precisions:
        for d in DEPTHS:
            lines.append(f"| {p} | {d} | " + " | ".join(f"{pc[p, d, lab]['correct']}/{pc[p, d, lab]['n']}" for lab in LABELS) + " |")
    lines += ["", "## Automatic checks", ""] + ([f"- {f}" for f in out["flags"]] or ["- none triggered"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    runs, headers = {}, {}
    for p in PRECISIONS:
        path = args.runs / f"proofwriter_{p}.jsonl"
        if path.exists():
            headers[p], runs[p] = load_run(path)
    assert "bf16" in runs, "BF16 baseline run is required"
    shared = {k for k in ("eval_set_sha256", "prompts_sha256", "decoding")}
    for p in runs:
        assert all(headers[p][k] == headers["bf16"][k] for k in shared), f"{p} provenance differs from bf16"
    out = analyze(runs)
    out["provenance"] = {p: headers[p] for p in runs}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    for name in ("accuracy", "penalty", "per_class", "pred_share"):
        write_csv(args.out_dir / f"{name}.csv", out[name])
    (args.out_dir / "summary.md").write_text(summary_md(out))
    plot(out, args.out_dir / "accuracy_vs_depth.png")
    print(summary_md(out))


if __name__ == "__main__":
    main()
