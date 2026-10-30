"""Thinking-mode analysis: accuracy by depth per precision, paired penalties, depth slope,
plus how long the model thinks and how often it runs out of budget.

Reuses analyze.analyze (Wilson CIs, paired bootstrap within depth, slope of Δ on depth).
TRUNCATED answers (budget hit before </think>) count as wrong, as in the runner.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import analyze
from proofwriter_test import DEPTHS, LABELS, ROOT
from thinking import THINK_END

ORDER = ["bf16", "8bit", "6bit", "5bit", "4bit", "3bit", "gptq4"]


def load(path):
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return lines[0], {row["index"]: row for row in lines[1:]}


def lenient_prediction(output):
    """Secondary scoring: first standalone label word after </think> (accepts "Answer: Unknown", "**True**").

    The strict parser (primary, pre-registered) requires the answer to *start* with the label.
    """
    if THINK_END not in output:
        return "TRUNCATED"
    match = re.search(r"\b(True|False|Unknown)\b", output.split(THINK_END, 1)[1])
    return match.group(1) if match else "INVALID"


def rescore_lenient(runs):
    out = {}
    for p, run in runs.items():
        out[p] = {}
        for i, r in run.items():
            pred = lenient_prediction(r["output"])
            out[p][i] = dict(r, prediction=pred, correct=pred == r["gold"])
    return out


def thinking_stats(runs, common):
    rows = []
    for p, run in runs.items():
        for d in DEPTHS:
            items = [run[i] for i in common if run[i]["depth"] == d]
            rows.append(dict(precision=p, depth=d, n=len(items),
                             mean_tokens=float(np.mean([r["generated_tokens"] for r in items])),
                             truncated=sum(r["prediction"] == "TRUNCATED" for r in items)))
    return rows


def lenient_md(out, precisions):
    acc = {(r["precision"], r["depth"]): r for r in out["accuracy"]}
    lines = ["", "## Sensitivity: lenient answer parsing (first True/False/Unknown word after </think>)", "",
             "| Depth | " + " | ".join(precisions) + " |", "|---|" + "---|" * len(precisions)]
    for d in DEPTHS:
        lines.append(f"| {d} | " + " | ".join(f"{acc[p, d]['acc'] * 100:.0f}" for p in precisions) + " |")
    lines += ["", "| Precision | Overall Δ (pts) | Slope of Δ per depth (pts) |", "|---|---|---|"]
    for t in out["trend"]:
        lines.append(f"| {t['precision']} | {t['overall_delta'] * 100:+.1f} [{t['overall_ci'][0] * 100:+.1f}, {t['overall_ci'][1] * 100:+.1f}] "
                     f"| {t['slope_per_depth'] * 100:+.2f} [{t['slope_ci'][0] * 100:+.2f}, {t['slope_ci'][1] * 100:+.2f}] |")
    return "\n".join(lines) + "\n"


def finish_md(runs, common, precisions):
    """Is the loss from wrong reasoning or from failing to finish? Accuracy on items both finished."""
    base = runs["bf16"]
    lines = ["", "## Wrong reasoning or failure to finish?", "",
             "Accuracy on the items where the quantized model closed </think> with a parseable answer "
             "(strict parser), compared with BF16 on the same items.", "",
             "| Precision | Items finished | Acc (quantized) | Acc (BF16, same items) | Truncated by depth 0→5 |",
             "|---|---|---|---|---|"]
    for p in precisions:
        if p == "bf16":
            continue
        run = runs[p]
        fin = [i for i in common if run[i]["prediction"] in LABELS]
        trunc = [sum(run[i]["prediction"] == "TRUNCATED" for i in common if run[i]["depth"] == d) for d in DEPTHS]
        if fin:
            qa = sum(run[i]["correct"] for i in fin) / len(fin)
            ba = sum(base[i]["correct"] for i in fin) / len(fin)
            lines.append(f"| {p} | {len(fin)} | {qa:.0%} | {ba:.0%} | {' / '.join(map(str, trunc))} |")
        else:
            lines.append(f"| {p} | 0 | — | — | {' / '.join(map(str, trunc))} |")
    return "\n".join(lines) + "\n"


def summary_md(family, out, stats, precisions):
    acc = {(r["precision"], r["depth"]): r for r in out["accuracy"]}
    pen = {(r["precision"], r["depth"]): r for r in out["penalty"]}
    st = {(r["precision"], r["depth"]): r for r in stats}
    quant = [p for p in precisions if p != "bf16"]
    n_depth = out["n_items"] // len(DEPTHS)
    lines = [f"# Thinking-mode precision sweep — {family}", "",
             f"{out['n_items']} paired items ({n_depth} per depth, balanced across labels). Greedy decoding, "
             "4,096-token budget; TRUNCATED counts as wrong. Accuracy CIs: Wilson 95%. "
             f"Penalty CIs: paired bootstrap, {analyze.BOOT:,} resamples within depth.", "",
             "## Accuracy by depth, % [95% CI]", "",
             "| Depth | " + " | ".join(precisions) + " |", "|---|" + "---|" * len(precisions)]
    for d in DEPTHS:
        lines.append(f"| {d} | " + " | ".join(
            f"{acc[p, d]['acc'] * 100:.0f} [{acc[p, d]['lo'] * 100:.0f}, {acc[p, d]['hi'] * 100:.0f}]" for p in precisions) + " |")
    lines += ["", "## Penalty Δ = A(BF16) − A(b), points [95% CI]", "",
              "| Depth | " + " | ".join(quant) + " |", "|---|" + "---|" * len(quant)]
    for d in DEPTHS:
        lines.append(f"| {d} | " + " | ".join(
            f"{pen[p, d]['delta'] * 100:+.1f} [{pen[p, d]['lo'] * 100:+.1f}, {pen[p, d]['hi'] * 100:+.1f}]" for p in quant) + " |")
    lines += ["", "## Does the penalty grow with depth?", "",
              "| Precision | Overall Δ (pts) | Slope of Δ per depth (pts) | Δ(d5) − Δ(d0) (pts) |", "|---|---|---|---|"]
    for t in out["trend"]:
        lines.append(f"| {t['precision']} | {t['overall_delta'] * 100:+.1f} [{t['overall_ci'][0] * 100:+.1f}, {t['overall_ci'][1] * 100:+.1f}] "
                     f"| {t['slope_per_depth'] * 100:+.2f} [{t['slope_ci'][0] * 100:+.2f}, {t['slope_ci'][1] * 100:+.2f}] "
                     f"| {t['delta_d5_minus_d0'] * 100:+.1f} [{t['gap_ci'][0] * 100:+.1f}, {t['gap_ci'][1] * 100:+.1f}] |")
    lines += ["", "## Thinking length (mean generated tokens) and truncations", "",
              "| Depth | " + " | ".join(precisions) + " |", "|---|" + "---|" * len(precisions)]
    for d in DEPTHS:
        lines.append(f"| {d} | " + " | ".join(f"{st[p, d]['mean_tokens']:.0f} ({st[p, d]['truncated']})" for p in precisions) + " |")
    lines += ["", "## Automatic checks", ""] + ([f"- {f}" for f in out["flags"]] or ["- none triggered"])
    return "\n".join(lines) + "\n"


def plot(out, path, family, precisions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(100 / 3, color="#8a8984", lw=1, ls=(0, (3, 3)))
    ax.text(-0.35, 100 / 3 + 1.5, "chance (33%)", fontsize=8, color=muted)
    offsets = np.linspace(-0.1, 0.1, len(precisions))
    for color, off, p in zip(palette, offsets, precisions):
        rows = [r for r in out["accuracy"] if r["precision"] == p]
        x = np.array([r["depth"] for r in rows]) + off
        y = np.array([r["acc"] for r in rows]) * 100
        err = np.array([[r["acc"] - r["lo"], r["hi"] - r["acc"]] for r in rows]).T * 100
        ax.errorbar(x, y, yerr=err, color=color, lw=2, marker="o", ms=5, elinewidth=1.2, capsize=0,
                    markeredgecolor=surface, markeredgewidth=1.5, label=p.upper() if p == "bf16" else p)
    ax.set_xticks(DEPTHS)
    ax.set_ylim(0, 100)
    ax.set_xlabel("ProofWriter reasoning depth", fontsize=9, color=muted)
    ax.set_ylabel("Accuracy (%)", fontsize=9, color=muted)
    ax.set_title(f"{family} with step-by-step thinking: accuracy vs depth by precision\n"
                 f"{out['n_items'] // len(DEPTHS)} items per depth, 95% Wilson CIs", fontsize=10, color=ink, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="qwen3-0.6b")
    args = parser.parse_args()
    runs_dir = ROOT / "results" / "thinking" / "raw" / args.family
    runs, headers = {}, {}
    for p in ORDER:
        path = runs_dir / f"proofwriter_{p}.jsonl"
        if path.exists():
            headers[p], runs[p] = load(path)
    assert "bf16" in runs, "BF16 baseline run is required"
    for p in runs:
        for k in ("eval_set_sha256", "prompts_sha256", "max_tokens", "chat_template_kwargs", "decoding"):
            assert headers[p][k] == headers["bf16"][k], f"{p} {k} differs from bf16"
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    counts = {d: sum(runs["bf16"][i]["depth"] == d for i in common) for d in DEPTHS}
    assert len(set(counts.values())) == 1, f"unbalanced depths in the common items: {counts}"
    runs = {p: {i: run[i] for i in common} for p, run in runs.items()}
    precisions = list(runs)
    out = analyze.analyze(runs)
    stats = thinking_stats(runs, common)
    out_dir = ROOT / "analysis" / "thinking" / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    lenient = analyze.analyze(rescore_lenient(runs))
    md = (summary_md(args.family, out, stats, precisions) + finish_md(runs, common, precisions)
          + lenient_md(lenient, precisions))
    (out_dir / "analysis.json").write_text(json.dumps(dict(out, thinking=stats, lenient=lenient, provenance=headers), indent=2) + "\n")
    (out_dir / "summary.md").write_text(md)
    plot(out, out_dir / "accuracy_vs_depth.png", args.family, precisions)
    print(md)


if __name__ == "__main__":
    main()
