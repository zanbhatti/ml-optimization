"""Turn the speed/memory benchmark into a table and a chart, and test the bandwidth explanation.

Decoding one token has to read every weight once (and the KV cache), but does only ~2
floating-point operations per weight. The GPU can do far more arithmetic than that per byte
it can fetch, so decode speed is capped by memory bandwidth:

    decode tok/s  <=  bandwidth / bytes read per token  ~=  bandwidth / weight size

Halving the bytes per weight should roughly double decode speed. Prefill processes L prompt
tokens per pass, so each weight fetched is reused L times: compute-bound, and smaller weights
help little (dequantizing can even cost a bit). The table reports each model's measured
decode speed as a fraction of this bandwidth ceiling.
"""
import argparse
import csv
import json
from pathlib import Path

from proofwriter_test import ROOT

BANDWIDTH_GBS = 150.0  # Apple M3 Pro spec (unified memory)
ORDER = ["bf16", "8bit", "6bit", "5bit", "4bit", "3bit", "gptq4"]
FAMILY_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-4b-instruct-2507"]


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (FAMILY_ORDER.index(r["family"]), ORDER.index(r["precision"])))
    bf16 = {r["family"]: r for r in rows if r["precision"] == "bf16"}
    for r in rows:
        r["ceiling_tps"] = BANDWIDTH_GBS / r["disk_gb"]
        base = bf16.get(r["family"])
        if base:
            r["bits_per_weight"] = 16 * r["disk_gb"] / base["disk_gb"]
            for ctx in (512, 4096):
                key = f"decode_tps_{ctx}"
                if key in r:
                    r[f"decode_speedup_{ctx}"] = r[key] / base[key]
    return rows


def label(family):
    return family.replace("qwen3-", "Qwen3-").replace("b", "B").replace("-instruct-2507", "-Instruct")


def summary_md(rows):
    out = ["# Speed and memory by size and precision (Apple M3 Pro, 18 GB, MLX)", "",
           f"Decode = generating tokens one at a time; prefill = reading the prompt. Medians of "
           f"{rows[0]['repeats']} runs, {rows[0]['decode_tokens']} decoded tokens each. "
           f"Ceiling = {BANDWIDTH_GBS:.0f} GB/s ÷ weight size: the fastest decode could go if it did nothing "
           f"but read every weight once per token.", "",
           "| Model | Precision | Weights (GB) | Bits/weight | Peak mem @4k (GB) | Prefill tok/s @512 | Prefill tok/s @4k "
           "| Decode tok/s @512 | Decode tok/s @4k | Decode speedup vs BF16 | Ceiling tok/s | % of ceiling |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        get = lambda k, f="{:.0f}": f.format(r[k]) if k in r else "—"
        out.append(f"| {label(r['family'])} | {r['precision']} | {r['disk_gb']:.2f} | {get('bits_per_weight', '{:.1f}')} "
                   f"| {get('peak_gb_4096', '{:.2f}')} | {get('prefill_tps_512')} | {get('prefill_tps_4096')} "
                   f"| {get('decode_tps_512')} | {get('decode_tps_4096')} | {get('decode_speedup_512', '{:.2f}×')} "
                   f"| {r['ceiling_tps']:.0f} | {100 * r['decode_tps_512'] / r['ceiling_tps']:.0f}% |")
    return "\n".join(out) + "\n"


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]
    families = list(dict.fromkeys(r["family"] for r in rows))
    ctx = 4096 if all("prefill_tps_4096" in r for r in rows) else 512
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    sizes = [r["disk_gb"] for r in rows]
    xs = [min(sizes) * 0.8, max(sizes) * 1.25]
    left.plot(xs, [BANDWIDTH_GBS / x for x in xs], color=muted, lw=1, ls="--")
    left.text(xs[0] * 1.05, BANDWIDTH_GBS / xs[0] * 0.8, f"bandwidth ceiling\n{BANDWIDTH_GBS:.0f} GB/s ÷ size",
              fontsize=7, color=muted, va="top")
    for color, family in zip(colors, families):
        mine = [r for r in rows if r["family"] == family]
        left.plot([r["disk_gb"] for r in mine], [r["decode_tps_512"] for r in mine], color=color, lw=1.5,
                  marker="o", ms=5, markeredgecolor=surface, markeredgewidth=1.5, label=label(family))
        for r in mine:
            left.annotate(r["precision"], (r["disk_gb"], r["decode_tps_512"]), xytext=(4, 4),
                          textcoords="offset points", fontsize=6, color=muted)
        right.plot([ORDER.index(r["precision"]) for r in mine], [r[f"prefill_tps_{ctx}"] for r in mine],
                   color=color, lw=1.5, marker="o", ms=5, markeredgecolor=surface, markeredgewidth=1.5,
                   label=label(family))
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Weight size (GB, log scale)", fontsize=9, color=muted)
    left.set_ylabel("Decode tokens/s (log scale)", fontsize=9, color=muted)
    left.set_title("Decode: smaller weights → faster, tracking bandwidth", fontsize=10, color=ink, loc="left")
    used = sorted({ORDER.index(r["precision"]) for r in rows})
    right.set_xticks(used, [ORDER[i] for i in used])
    right.set_yscale("log")
    right.set_ylabel(f"Prefill tokens/s @{ctx} context (log scale)", fontsize=9, color=muted)
    right.set_title("Prefill: compute-bound, precision barely matters", fontsize=10, color=ink, loc="left")
    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(color=grid, lw=0.8, which="major")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=muted, labelsize=8)
    right.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "benchmark" / "m3pro-18gb.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis" / "benchmark")
    args = parser.parse_args()
    rows = load(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(summary_md(rows))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.out / "speed_vs_size.png")
    print(summary_md(rows))


if __name__ == "__main__":
    main()
