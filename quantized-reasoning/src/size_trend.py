"""Cross-size summary of per-family forced-choice analyses (run analyze_forced_choice.py first)."""
import argparse
import json
from pathlib import Path

from analyze_forced_choice import ORDER
from proofwriter_test import DEPTHS, ROOT

SIZES = {"qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7, "qwen3-4b": 4.0}


def load(family):
    return json.loads((ROOT / "analysis" / "forced_choice" / family / "analysis.json").read_text())


def summarize(families):
    rows = []
    for family in families:
        a = load(family)
        by = {(r["precision"], r["depth"]): r for r in a["by_depth"] if r["subset"] == "all items"}
        trends = {t["precision"]: t for t in a["trends"] if t["subset"] == "all items"}
        conf = {r["precision"]: r for r in a.get("confidence_control", [])}
        for p in ORDER:
            if (p, 0) not in by:
                continue
            deep = [by[p, d] for d in DEPTHS if d > 0]
            row = dict(family=family, size_b=SIZES.get(family), precision=p,
                       acc_d0=by[p, 0]["acc"], acc_d1_5=sum(r["acc"] for r in deep) / len(deep),
                       flip_rate=sum(by[p, d]["flip_rate"] for d in DEPTHS) / len(DEPTHS))
            if p in trends:
                t = trends[p]
                row.update(d_margin=t["overall_d_margin"], d_margin_ci=t["overall_d_margin_ci"],
                           slope_d_margin=t["slope_d_margin"], slope_d_margin_ci=t["slope_d_margin_ci"])
            if p in conf:
                row.update(deep_effect=conf[p]["deep_effect"], deep_ci=conf[p]["deep_ci"],
                           share={lab: conf[p][lab] for lab in ("True", "False", "Unknown")})
            elif p == "bf16":
                base = [json.loads(line)["prediction"] for line in
                        (ROOT / "results" / "forced_choice" / "raw" / family / "proofwriter_bf16.jsonl").read_text().splitlines()[1:]]
                row.update(share={lab: base.count(lab) / len(base) for lab in ("True", "False", "Unknown")})
            rows.append(row)
    return rows


def md(rows):
    families = list(dict.fromkeys(r["family"] for r in rows))
    get = {(r["family"], r["precision"]): r for r in rows}
    out = ["# Qwen3 size trend — forced-choice ProofWriter (540 paired items per model)", "",
           "## BF16 baseline: is there headroom beyond depth 0?", "",
           "| Model | Acc depth 0 | Mean acc depths 1–5 | (chance 33%) |", "|---|---|---|---|"]
    for f in families:
        r = get[f, "bf16"]
        out.append(f"| {f} | {r['acc_d0']:.1%} | {r['acc_d1_5']:.1%} | |")
    quant = [p for p in ORDER if p != "bf16"]
    out += ["", "## Answers changed vs BF16 (flip rate)", "", "| Model | " + " | ".join(quant) + " |",
            "|---|" + "---|" * len(quant)]
    for f in families:
        out.append(f"| {f} | " + " | ".join(f"{get[f, p]['flip_rate']:.1%}" if (f, p) in get else "—" for p in quant) + " |")
    out += ["", "## Margin loss vs BF16 (nats, [95% CI])", "", "| Model | " + " | ".join(quant) + " |",
            "|---|" + "---|" * len(quant)]
    for f in families:
        out.append(f"| {f} | " + " | ".join(
            f"{get[f, p]['d_margin']:+.2f} [{get[f, p]['d_margin_ci'][0]:+.2f}, {get[f, p]['d_margin_ci'][1]:+.2f}]"
            if (f, p) in get else "—" for p in quant) + " |")
    out += ["", "## Depth effect on flips after controlling for BF16 confidence (pts, [95% CI])", "",
            "Positive = deep items flip more than equally-confident shallow items.", "",
            "| Model | " + " | ".join(quant) + " |", "|---|" + "---|" * len(quant)]
    for f in families:
        out.append(f"| {f} | " + " | ".join(
            f"{get[f, p]['deep_effect'] * 100:+.1f} [{get[f, p]['deep_ci'][0] * 100:+.1f}, {get[f, p]['deep_ci'][1] * 100:+.1f}]"
            if (f, p) in get and "deep_effect" in get[f, p] else "—" for p in quant) + " |")
    out += ["", "## Predicted-label shares (gold is 1/3 each): quantization shifts the label prior", "",
            "| Model | " + " | ".join(ORDER) + " |", "|---|" + "---|" * len(ORDER)]
    for f in families:
        out.append(f"| {f} | " + " | ".join(
            "T{True:.0%} F{False:.0%} U{Unknown:.0%}".format(**get[f, p]["share"]) if (f, p) in get and "share" in get[f, p] else "—"
            for p in ORDER) + " |")
    return "\n".join(out) + "\n"


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    bits = {"8bit": 8, "6bit": 6, "5bit": 5, "4bit": 4, "3bit": 3}
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for color, family in zip(colors, dict.fromkeys(r["family"] for r in rows)):
        pts = [(bits[r["precision"]], r["flip_rate"] * 100) for r in rows if r["family"] == family and r["precision"] in bits]
        x, y = zip(*pts)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, markeredgecolor=surface, markeredgewidth=1.5,
                label=family.replace("qwen3-", "Qwen3-").replace("b", "B"))
    ax.set_xticks(list(bits.values()))
    ax.invert_xaxis()
    ax.set_ylim(0, None)
    ax.set_xlabel("Bits per weight (MLX affine, group size 64)", fontsize=9, color=muted)
    ax.set_ylabel("Answers changed vs BF16 (%)", fontsize=9, color=muted)
    ax.set_title("Quantization sensitivity by model size\nQwen3, ProofWriter forced-choice, 540 paired items",
                 fontsize=10, color=ink, loc="left")
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
    parser.add_argument("families", nargs="*", default=list(SIZES))
    args = parser.parse_args()
    rows = summarize(args.families)
    out = ROOT / "analysis" / "forced_choice" / "size_trend"
    out.mkdir(parents=True, exist_ok=True)
    (out / "size_trend.json").write_text(json.dumps(rows, indent=2) + "\n")
    text = md(rows)
    (out / "summary.md").write_text(text)
    plot(rows, out / "flip_rate_vs_bits.png")
    print(text)


if __name__ == "__main__":
    main()
