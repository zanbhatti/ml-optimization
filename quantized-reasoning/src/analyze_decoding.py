"""Is 4-bit's thinking-mode loss an artifact of greedy decoding or the 4,096-token budget?

Greedy decoding (always take the most likely token) is known to fall into repetition loops,
and a quantized model might loop more without reasoning any worse. Two checks on the same
180 items, compared against the original greedy/4,096 runs:

  sampled  - temp 0.6, top-p 0.95, top-k 20 (Qwen3's recommended thinking settings), 2 seeds;
             per item, correctness is averaged over seeds (0, 0.5 or 1).
  8k       - greedy with an 8,192-token budget. Only items truncated at 4,096 were rerun;
             greedy is deterministic, so every other item's output is unchanged.

If the 4-bit penalty (Δ = BF16 − 4-bit accuracy) and its growth with depth survive both, it
is not a decoding artifact.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import bootstrap_penalties, ci, load_run, SEED
from analyze_thinking import lenient_prediction
from proofwriter_test import DEPTHS, ROOT
from thinking import TRUNCATED

SAMPLING = "_t0.6_p0.95_k20_s{seed}"
SEEDS = (1, 2)


def merged_8k(raw, precision):
    """Greedy/4096 records with truncated items replaced by their 8,192-token rerun."""
    base = load_run(raw / f"proofwriter_{precision}.jsonl")[1]
    rerun = raw / f"proofwriter_{precision}_max8192_retrunc.jsonl"
    if not rerun.exists():
        return None
    redo = load_run(rerun)[1]
    truncated = {i for i, r in base.items() if r["prediction"] == TRUNCATED}
    if not truncated <= set(redo):
        return None  # rerun not finished
    return {i: redo.get(i, r) for i, r in base.items()}


def sampled(raw, precision):
    """Seed-averaged records: 'correct' and 'truncated' become fractions over seeds."""
    seeds = []
    for seed in SEEDS:
        path = raw / f"proofwriter_{precision}{SAMPLING.format(seed=seed)}.jsonl"
        if not path.exists():
            return None
        seeds.append(load_run(path)[1])
    common = set.intersection(*(set(s) for s in seeds))
    out = {}
    for i in common:
        rows = [s[i] for s in seeds]
        out[i] = dict(rows[0], correct=np.mean([r["correct"] for r in rows]),
                      truncated=np.mean([r["prediction"] == TRUNCATED for r in rows]),
                      lenient_correct=np.mean([lenient_prediction(r["output"]) == r["gold"] for r in rows]))
    return out


def conditions(raw, precisions):
    out = {"greedy, 4k tokens": {p: load_run(raw / f"proofwriter_{p}.jsonl")[1] for p in precisions
                                 if (raw / f"proofwriter_{p}.jsonl").exists()},
           "greedy, 8k tokens": {p: merged_8k(raw, p) for p in precisions},
           "sampled (2 seeds), 4k tokens": {p: sampled(raw, p) for p in precisions}}
    return {name: {p: recs for p, recs in runs.items() if recs} for name, runs in out.items()}


def summarize(runs, lenient=False):
    """Accuracy and truncation by depth per precision; Δ vs BF16 with paired bootstrap CIs."""
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    depths = np.array([runs["bf16"][i]["depth"] for i in common])

    def score(r):
        if lenient:
            return r.get("lenient_correct", float(lenient_prediction(r["output"]) == r["gold"]))
        return float(r["correct"])
    correct = {p: np.array([score(runs[p][i]) for i in common]) for p in runs}
    trunc = {p: np.array([float(runs[p][i].get("truncated", runs[p][i]["prediction"] == TRUNCATED))
                          for i in common]) for p in runs}
    out = {"n": len(common), "acc": {}, "trunc": {}, "delta": {}}
    for p in runs:
        out["acc"][p] = [float(correct[p][depths == d].mean()) for d in DEPTHS]
        out["trunc"][p] = [float(trunc[p][depths == d].mean()) for d in DEPTHS]
    draws = bootstrap_penalties(depths, correct, np.random.default_rng(SEED))
    xc = np.array(DEPTHS, dtype=float) - np.mean(DEPTHS)
    for p, D in draws.items():
        point = np.array([correct["bf16"][depths == d].mean() - correct[p][depths == d].mean() for d in DEPTHS])
        out["delta"][p] = {"by_depth": point.tolist(), "by_depth_ci": [ci(D[:, j]) for j in range(len(DEPTHS))],
                           "overall": float(point.mean()), "overall_ci": ci(D.mean(axis=1)),
                           "slope": float((xc * point).sum() / (xc ** 2).sum()),
                           "slope_ci": ci((D * xc).sum(axis=1) / (xc ** 2).sum())}
    return out


def pct(x):
    return f"{100 * x:.0f}"


def summary_md(results):
    lines = ["# Greedy-decoding check — Qwen3-0.6B thinking mode", "",
             "Δ = BF16 accuracy − quantized accuracy (points; positive = quantization hurts). 95% CIs from a "
             "paired bootstrap within depth (10,000 resamples). Slope = change in Δ per extra reasoning step.", ""]
    for scoring in ("strict", "lenient"):
        lines += [f"## Penalty under each decoding setup ({scoring} parser)", "",
                  "| Setup | Precision | Items | Δ overall | Slope per depth | Δ by depth 0–5 |", "|---|---|---|---|---|---|"]
        for name, res in results.items():
            r = res[scoring]
            for p, d in r["delta"].items():
                lines.append(f"| {name} | {p} | {r['n']} | {100 * d['overall']:+.1f} [{100 * d['overall_ci'][0]:+.1f}, "
                             f"{100 * d['overall_ci'][1]:+.1f}] | {100 * d['slope']:+.2f} [{100 * d['slope_ci'][0]:+.2f}, "
                             f"{100 * d['slope_ci'][1]:+.2f}] | " + " / ".join(f"{100 * x:+.0f}" for x in d["by_depth"]) + " |")
        lines.append("")
    lines += ["## Accuracy and truncation by depth (strict parser)", "",
              "| Setup | Precision | Accuracy % by depth 0–5 | Truncated % by depth 0–5 | Truncated overall |",
              "|---|---|---|---|---|"]
    for name, res in results.items():
        r = res["strict"]
        for p in r["acc"]:
            lines.append(f"| {name} | {p} | " + " / ".join(pct(x) for x in r["acc"][p]) + " | "
                         + " / ".join(pct(x) for x in r["trunc"][p]) + f" | {pct(np.mean(r['trunc'][p]))}% |")
    return "\n".join(lines) + "\n"


def plot(results, path, precision="4bit"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(0, color="#c3c2b7", lw=1)
    for k, (color, (name, res)) in enumerate(zip(colors, results.items())):
        d = res["strict"]["delta"].get(precision)
        if not d:
            continue
        x = np.array(DEPTHS) + (k - 1) * 0.08
        y = 100 * np.array(d["by_depth"])
        lo, hi = (100 * np.array(v) for v in zip(*d["by_depth_ci"]))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, lw=2, marker="o", ms=5, capsize=2, elinewidth=1,
                    markeredgecolor=surface, label=f"{name}: Δ {100 * d['overall']:+.1f}")
    ax.set_xticks(DEPTHS)
    ax.set_xlabel("Reasoning depth (steps needed)", fontsize=9, color=muted)
    ax.set_ylabel(f"BF16 − {precision} accuracy (points)", fontsize=9, color=muted)
    ax.set_title(f"Does the 4-bit penalty survive non-greedy decoding and a bigger budget?\n"
                 f"Qwen3-0.6B thinking mode, ProofWriter, 95% CIs", fontsize=10, color=ink, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="qwen3-0.6b")
    parser.add_argument("--precisions", nargs="+", default=["bf16", "4bit", "gptq4"])
    args = parser.parse_args()
    raw = ROOT / "results" / "thinking" / "raw" / args.family
    results = {name: {"strict": summarize(runs), "lenient": summarize(runs, lenient=True)}
               for name, runs in conditions(raw, args.precisions).items() if "bf16" in runs and len(runs) > 1}
    out = ROOT / "analysis" / "thinking" / args.family
    out.mkdir(parents=True, exist_ok=True)
    (out / "decoding_check.md").write_text(summary_md(results))
    (out / "decoding_check.json").write_text(json.dumps(results, indent=1))
    plot(results, out / "decoding_check.png")
    print(summary_md(results))


if __name__ == "__main__":
    main()
