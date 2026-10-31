"""Check block-wise GPTQ (gptq4b) against mlx_lm's GPTQ (gptq4) on the same model.

Two direct comparisons of the quantized weights (the behavioural comparisons — forced-choice
flips/KL and thinking accuracy — reuse the normal runners and analyses):

  1. Integer levels: unpack every GPTQ layer's 4-bit integers and compare them position by
     position (max and mean difference in levels, % identical), plus scales and biases.
  2. Layer output error: feed each layer the inputs it sees in the full-precision model on
     calibration text and measure the relative error ||Ŵx − Wx||² / ||Wx||² for both versions.
     Same algorithm should give the same error (we ask for within 5% per layer).
"""
import argparse
import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from models import FAMILIES, GPTQ, model_path
from proofwriter_test import ROOT


def levels(layer):
    """Unpack a QuantizedLinear's uint32 words into its integer levels (0 .. 2^bits − 1)."""
    bits = layer.bits
    shifts = mx.arange(0, 32, bits, dtype=mx.uint32)
    q = (layer.weight[..., None] >> shifts) & (2**bits - 1)
    return q.reshape(layer.weight.shape[0], -1).astype(mx.int32)


def gptq_layers(model):
    return {k: l for k, l in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
            if isinstance(l, nn.QuantizedLinear)}


def compare_levels(a, b):
    out, total, same, worst, absdiff = {}, 0, 0, 0, 0
    for k in a:
        d = mx.abs(levels(a[k]) - levels(b[k]))
        n = d.size
        out[k] = {"identical": (d == 0).sum().item() / n, "max_level_diff": d.max().item(),
                  "scale_max_rel_diff": (mx.abs(a[k].scales - b[k].scales) /
                                         (mx.abs(a[k].scales) + 1e-12)).max().item()}
        total += n
        same += (d == 0).sum().item()
        absdiff += d.sum().item()
        worst = max(worst, out[k]["max_level_diff"])
    return {"identical": same / total, "mean_level_diff": absdiff / total, "max_level_diff": worst}, out


def output_errors(family, quantized, n_samples=16, seq=512):
    """Relative output error per GPTQ layer, on inputs from the full-precision forward pass."""
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from mlx_lm.quant.utils import load_data
    mx.random.seed(GPTQ["seed"])
    ref, tokenizer = load(snapshot_download(FAMILIES[family], local_files_only=True))
    data = load_data(tokenizer, n_samples, seq)
    names = list(gptq_layers(quantized[next(iter(quantized))]).keys())
    errs = {name: {p: [0.0, 0.0] for p in quantized} for name in names}  # [error energy, signal energy]
    ref_layers = dict(tree_flatten(ref.leaf_modules(), is_leaf=nn.Module.is_module))
    q_layers = {p: gptq_layers(m) for p, m in quantized.items()}

    class Probe(nn.Module):
        def __init__(self, name, module):
            super().__init__()
            self.module, self._name = module, name

        def __call__(self, x):
            y = self.module(x)
            for p in quantized:
                e = q_layers[p][self._name](x) - y
                acc = errs[self._name][p]
                acc[0] += mx.sum(e.astype(mx.float32) ** 2).item()
                acc[1] += mx.sum(y.astype(mx.float32) ** 2).item()
            return y

    from mlx.utils import tree_unflatten
    ref.update_modules(tree_unflatten([(n, Probe(n, ref_layers[n])) for n in names]))
    for s in range(0, len(data), 4):
        mx.eval(ref(data[s:s + 4]))
    return {n: {p: v[0] / v[1] for p, v in d.items()} for n, d in errs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="qwen3-0.6b")
    parser.add_argument("--a", default="gptq4", help="reference (mlx_lm)")
    parser.add_argument("--b", default="gptq4b", help="block-wise")
    args = parser.parse_args()
    from mlx_lm import load
    models = {p: load(model_path(args.family, p))[0] for p in (args.a, args.b)}
    la, lb = gptq_layers(models[args.a]), gptq_layers(models[args.b])
    assert la.keys() == lb.keys()
    summary, per_layer = compare_levels(la, lb)
    errors = output_errors(args.family, models)
    ratios = {n: e[args.b] / e[args.a] for n, e in errors.items()}
    within = sum(abs(r - 1) <= 0.05 for r in ratios.values())
    summary.update({"layers": len(ratios), "output_error_within_5pct": within,
                    "output_error_ratio_min": min(ratios.values()), "output_error_ratio_max": max(ratios.values()),
                    "mean_rel_error": {p: sum(e[p] for e in errors.values()) / len(errors) for p in (args.a, args.b)}})
    out = ROOT / "analysis" / "gptq_validation" / args.family
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps({"summary": summary, "per_layer": per_layer,
                                                     "output_errors": errors}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
