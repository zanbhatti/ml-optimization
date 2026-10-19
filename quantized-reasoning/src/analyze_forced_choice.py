"""Forced-choice analysis: dose-response over bit width and its interaction with depth.

Per item i and precision b (paired against BF16 on the same item):
  margin loss   ΔM_i = margin_i(bf16) − margin_i(b)              (nats)
  KL_i          KL(p_bf16 ‖ p_b) over the 3 label probabilities  (nats)
  flip_i        prediction differs from BF16
The depth interaction is the slope of ΔM (or Δaccuracy) on depth, with CIs from a
bootstrap that resamples items within each depth. Positive slope = the quantization
penalty grows with reasoning depth.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze import CHANCE, ci, wilson
from proofwriter_test import DEPTHS, LABELS, ROOT

ORDER = ["bf16", "8bit", "6bit", "5bit", "4bit", "3bit"]
BITS = {"8bit": 8, "6bit": 6, "5bit": 5, "4bit": 4, "3bit": 3}
BOOT = 10_000
SEED = 0
GROUPS = {"depth 0–1": (0, 1), "depth 2–3": (2, 3), "depth 4–5": (4, 5)}


def load(path):
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return lines[0], {row["index"]: row for row in lines[1:]}


def arrays(runs):
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    for i in common:
        assert len({runs[p][i]["key"] for p in runs}) == 1
    base = runs["bf16"]
    out = {"index": np.array(common), "depth": np.array([base[i]["depth"] for i in common]),
           "gold": np.array([base[i]["gold"] for i in common])}
    for p, run in runs.items():
        lp = np.array([[run[i]["logprobs"][lab] for lab in LABELS] for i in common])
        probs = np.exp(lp - lp.max(axis=1, keepdims=True))
        out[p] = {"lp": lp, "probs": probs / probs.sum(axis=1, keepdims=True),
                  "margin": np.array([run[i]["margin"] for i in common]),
                  "correct": np.array([run[i]["correct"] for i in common], dtype=float),
                  "pred": np.array([run[i]["prediction"] for i in common])}
    return out


def slope(x, y):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def stratified_boot(depths, rng):
    """(BOOT, n) index draws, resampling items within each depth."""
    blocks = []
    for d in DEPTHS:
        idx = np.flatnonzero(depths == d)
        blocks.append(idx[rng.integers(0, len(idx), size=(BOOT, len(idx)))])
    return np.concatenate(blocks, axis=1)


def boot_slopes(depths, values, draws):
    x = depths[draws].astype(float)
    y = values[draws]
    xc = x - x.mean(axis=1, keepdims=True)
    return (xc * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / (xc ** 2).sum(axis=1)


def analyze(data, subset_name="all items", mask=None):
    rng = np.random.default_rng(SEED)
    mask = np.ones(len(data["depth"]), bool) if mask is None else mask
    depths = data["depth"][mask]
    draws = stratified_boot(depths, rng)
    precisions = [p for p in ORDER if p in data]
    base = data["bf16"]
    rows, trends = [], []
    for p in precisions:
        d_ = data[p]
        dm = (base["margin"] - d_["margin"])[mask]
        dacc = (base["correct"] - d_["correct"])[mask]
        pb, pq = base["probs"][mask], d_["probs"][mask]
        kl = (pb * (np.log(pb + 1e-12) - np.log(pq + 1e-12))).sum(axis=1)
        flip = (base["pred"] != d_["pred"])[mask].astype(float)
        correct, margin = d_["correct"][mask], d_["margin"][mask]
        for d in DEPTHS:
            m = depths == d
            k, n = int(correct[m].sum()), int(m.sum())
            lo, hi = wilson(k, n)
            idx = np.flatnonzero(m)
            boot_idx = idx[rng.integers(0, len(idx), size=(BOOT, len(idx)))]
            rows.append(dict(subset=subset_name, precision=p, depth=d, n=n, acc=k / n, acc_lo=lo, acc_hi=hi,
                             mean_margin=float(margin[m].mean()),
                             d_acc=float(dacc[m].mean()), d_acc_ci=ci(dacc[boot_idx].mean(axis=1)),
                             d_margin=float(dm[m].mean()), d_margin_ci=ci(dm[boot_idx].mean(axis=1)),
                             kl=float(kl[m].mean()), flip_rate=float(flip[m].mean())))
        if p != "bf16":
            trends.append(dict(subset=subset_name, precision=p, bits=BITS[p],
                               overall_d_acc=float(dacc.mean()), overall_d_acc_ci=ci(dacc[draws].mean(axis=1)),
                               overall_d_margin=float(dm.mean()), overall_d_margin_ci=ci(dm[draws].mean(axis=1)),
                               slope_d_margin=slope(depths.astype(float), dm),
                               slope_d_margin_ci=ci(boot_slopes(depths, dm, draws)),
                               slope_d_acc=slope(depths.astype(float), dacc),
                               slope_d_acc_ci=ci(boot_slopes(depths, dacc, draws)),
                               slope_kl=slope(depths.astype(float), kl),
                               slope_kl_ci=ci(boot_slopes(depths, kl, draws))))
    return rows, trends


def confidence_control(data):
    """Is depth a predictor of flips once BF16's own confidence is held fixed?

    Linear probability model flip ~ 1 + gap + deep, where gap is BF16's top-1 minus
    top-2 label log-prob and deep = depth ≥ 1. Item bootstrap CI on the deep coefficient.
    """
    rng = np.random.default_rng(SEED)
    lp = data["bf16"]["lp"]
    top2 = np.sort(lp, axis=1)[:, ::-1]
    gap = top2[:, 0] - top2[:, 1]
    deep = (data["depth"] > 0).astype(float)
    X = np.c_[np.ones_like(gap), gap, deep]
    draws = rng.integers(0, len(gap), size=(2000, len(gap)))
    rows = []
    for p in ORDER[1:]:
        if p not in data:
            continue
        flip = (data[p]["pred"] != data["bf16"]["pred"]).astype(float)
        beta = np.linalg.lstsq(X, flip, rcond=None)[0][2]
        boots = [np.linalg.lstsq(X[s], flip[s], rcond=None)[0][2] for s in draws]
        share = {lab: float((data[p]["pred"] == lab).mean()) for lab in LABELS}
        rows.append(dict(precision=p, flip_rate=float(flip.mean()), deep_effect=float(beta), deep_ci=ci(boots), **share))
    return rows, {lab: float((data["bf16"]["pred"] == lab).mean()) for lab in LABELS}, \
        [float(np.median(gap[data["depth"] == d])) for d in DEPTHS]


def confidence_md(rows, base_share, gaps):
    out = ["", "## Depth vs BF16 confidence as the driver of answer flips", "",
           "BF16 median top-2 label gap (nats) by depth: " + ", ".join(f"d{d} {g:.1f}" for d, g in zip(DEPTHS, gaps)) + ".",
           "Deep effect = extra flip probability for depth ≥ 1 after controlling for that gap "
           "(linear probability model, 2,000 item bootstraps). Prediction shares show label-prior shifts "
           f"(BF16: True {base_share['True']:.0%}, False {base_share['False']:.0%}, Unknown {base_share['Unknown']:.0%}).", "",
           "| Precision | Flip rate | Deep effect (pts) | Pred. True | Pred. False | Pred. Unknown |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['precision']} | {r['flip_rate']:.1%} | {fmt_ci(r['deep_effect'], r['deep_ci'], 100, 1)} "
                   f"| {r['True']:.0%} | {r['False']:.0%} | {r['Unknown']:.0%} |")
    return "\n".join(out) + "\n"


def fmt_ci(point, interval, scale=1.0, digits=2, sign=True):
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"{f.format(point * scale)} [{f.format(interval[0] * scale)}, {f.format(interval[1] * scale)}]"


def summary_md(rows, trends, n_items):
    precisions = [p for p in ORDER if any(r["precision"] == p for r in rows)]
    get = {(r["subset"], r["precision"], r["depth"]): r for r in rows}
    out = ["# Forced-choice precision sweep — Qwen3-4B-Instruct-2507", "",
           f"{n_items} paired ProofWriter items; label chosen by argmax over the log-probs of the "
           "True/False/Unknown tokens at the first answer position (no format failures possible). "
           "Margin = log p(gold) − max log p(wrong), in nats. CIs: Wilson (accuracy) and paired "
           f"bootstrap with {BOOT:,} resamples within depth (everything else).", "",
           "## Accuracy by depth, %", "",
           "| Depth | " + " | ".join(precisions) + " |", "|---|" + "---|" * len(precisions)]
    for d in DEPTHS:
        out.append(f"| {d} | " + " | ".join(f"{get['all items', p, d]['acc'] * 100:.1f}" for p in precisions) + " |")
    out += ["", "## Dose-response and depth interaction (all items)", "",
            "Slope = change in the penalty per +1 depth. Positive with a CI above 0 means the penalty grows with depth.", "",
            "| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |",
            "|---|---|---|---|---|---|"]
    for subset in ("all items", "True/False items only"):
        if subset != "all items":
            out += ["", f"## Robustness: {subset} (depth is a proof length only for provable items)", "",
                    "| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |",
                    "|---|---|---|---|---|---|"]
        for t in (t for t in trends if t["subset"] == subset):
            out.append(f"| {t['precision']} | {fmt_ci(t['overall_d_acc'], t['overall_d_acc_ci'], 100, 1)} "
                       f"| {fmt_ci(t['overall_d_margin'], t['overall_d_margin_ci'])} "
                       f"| {fmt_ci(t['slope_d_acc'], t['slope_d_acc_ci'], 100, 2)} "
                       f"| {fmt_ci(t['slope_d_margin'], t['slope_d_margin_ci'], 1, 3)} "
                       f"| {fmt_ci(t['slope_kl'], t['slope_kl_ci'], 1, 4)} |")
    out += ["", "## Margin loss ΔM by depth (nats, [95% CI])", "",
            "| Depth | " + " | ".join(p for p in precisions if p != "bf16") + " |",
            "|---|" + "---|" * (len(precisions) - 1)]
    for d in DEPTHS:
        out.append(f"| {d} | " + " | ".join(fmt_ci(get['all items', p, d]['d_margin'], get['all items', p, d]['d_margin_ci'])
                                          for p in precisions if p != "bf16") + " |")
    out += ["", "## Flip rate vs BF16 by depth (% of items whose answer changes)", "",
            "| Depth | " + " | ".join(p for p in precisions if p != "bf16") + " |",
            "|---|" + "---|" * (len(precisions) - 1)]
    for d in DEPTHS:
        out.append(f"| {d} | " + " | ".join(f"{get['all items', p, d]['flip_rate'] * 100:.1f}"
                                          for p in precisions if p != "bf16") + " |")
    return "\n".join(out) + "\n"


def plot(rows, trends, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    quant = [p for p in ORDER if p != "bf16" and any(r["precision"] == p for r in rows)]
    bits = [BITS[p] for p in quant]
    get = {(r["precision"], r["depth"]): r for r in rows if r["subset"] == "all items"}
    for color, (name, ds) in zip(colors, GROUPS.items()):
        y = [np.mean([get[p, d]["d_margin"] for d in ds]) for p in quant]
        a1.plot(bits, y, color=color, lw=2, marker="o", ms=5, markeredgecolor=surface, markeredgewidth=1.5, label=name)
    a1.axhline(0, color=muted, lw=0.8)
    a1.set_xticks(bits)
    a1.invert_xaxis()
    a1.set_xlabel("Bits per weight (MLX affine, group size 64)", fontsize=9, color=muted)
    a1.set_ylabel("Margin loss vs BF16 (nats)", fontsize=9, color=muted)
    a1.set_title("Penalty vs bit width, by depth group", fontsize=10, color=ink, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    allt = [t for t in trends if t["subset"] == "all items"]
    x = np.arange(len(allt))
    s = np.array([t["slope_d_margin"] for t in allt])
    err = np.array([[t["slope_d_margin"] - t["slope_d_margin_ci"][0], t["slope_d_margin_ci"][1] - t["slope_d_margin"]] for t in allt]).T
    a2.errorbar(x, s, yerr=err, fmt="o", color=colors[0], ms=6, elinewidth=1.5, capsize=0,
                markeredgecolor=surface, markeredgewidth=1.5)
    a2.axhline(0, color=muted, lw=0.8)
    a2.set_xticks(x, [t["precision"] for t in allt])
    a2.set_ylabel("Slope of margin loss per +1 depth (nats)", fontsize=9, color=muted)
    a2.set_title("Does the penalty grow with depth? (95% CI)", fontsize=10, color=ink, loc="left")
    for ax in (a1, a2):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=muted, labelsize=8)
    fig.suptitle("Qwen3-4B-Instruct on ProofWriter, forced-choice scoring, 540 paired items",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "results" / "forced_choice" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "forced_choice")
    args = parser.parse_args()
    runs, headers = {}, {}
    for p in ORDER:
        path = args.runs / f"proofwriter_{p}.jsonl"
        if path.exists():
            headers[p], runs[p] = load(path)
    assert "bf16" in runs, "BF16 baseline run is required"
    for p in runs:
        for k in ("eval_set_sha256", "prompts_sha256", "scoring", "label_token_ids"):
            assert headers[p][k] == headers["bf16"][k], f"{p} {k} differs from bf16"
    data = arrays(runs)
    rows, trends = analyze(data)
    tf_rows, tf_trends = analyze(data, "True/False items only", data["gold"] != "Unknown")
    rows, trends = rows + tf_rows, trends + tf_trends
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(json.dumps(
        {"n_items": len(data["depth"]), "bootstrap": {"resamples": BOOT, "seed": SEED},
         "by_depth": rows, "trends": trends, "provenance": headers}, indent=2) + "\n")
    for name, table in (("by_depth", rows), ("trends", trends)):
        with (args.out_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    conf_rows, base_share, gaps = confidence_control(data)
    md = summary_md(rows, trends, len(data["depth"])) + confidence_md(conf_rows, base_share, gaps)
    (args.out_dir / "summary.md").write_text(md)
    plot(rows, trends, args.out_dir / "penalty_vs_bits.png")
    print(md)


if __name__ == "__main__":
    main()
