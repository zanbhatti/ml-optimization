"""Block-by-block GPTQ: the same result as mlx_lm.quant.gptq, in a fraction of the memory.

mlx_lm's GPTQ runs the whole model on the calibration text once and, along the way, builds a
Hessian H = sum of x xᵀ (x = the inputs a layer sees) for *every* Linear layer at once. Each H
is d_in × d_in in fp32, so the 4B model needs ~19 GB just for them — more than this Mac has.

This version walks through the model one transformer block at a time:

  1. Keep the calibration hidden states X entering the current block (one copy, ~0.5 GB).
  2. Run that block alone on X with Catchers on its Linears. Linears that read the same input
     share one H (q/k/v share one; gate/up share one; o_proj and down_proj get their own),
     so a block holds 4 Hessians instead of 7.
  3. Quantize the block's Linears with exactly mlx_lm's per-column GPTQ math.
  4. The block's output, computed with its ORIGINAL weights in step 2, becomes the next X.
     That reproduces mlx_lm, whose Hessians all come from one full-precision pass.
  5. Free the block's Hessians and move on.

`sequential=True` instead feeds the next block the output of the already-quantized block (as in
the GPTQ paper), so later layers calibrate on the errors earlier ones introduce. Not used yet.

Everything after the blocks matches mlx_lm: an untied lm_head (if any) gets GPTQ from the final
normed hidden states; other quantizable modules (Qwen3's tied embedding) use RTN fallback.
"""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.quant.gptq import quantize as pack
from mlx_lm.quant.utils import load_data
from mlx_lm.utils import compute_bits_per_weight, load, save

GPTQ_TYPES = (nn.Linear, SwitchLinear)
# Linears inside a block that read the identical input tensor, so their Hessians are identical.
SHARED_INPUTS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj"))


class HessianStore:
    """Plain holder (not a dict) so MLX doesn't treat the Hessians as module parameters."""

    def __init__(self):
        self.H = {}


class Catcher(nn.Module):
    """Wraps a Linear; if it leads its input group, adds xᵀx of every input into store.H[key]."""

    def __init__(self, module, store, key, leader):
        super().__init__()
        self.module = module
        self._store, self._key, self._leader = store, key, leader

    def __call__(self, x, *args, **kwargs):
        if self._leader:
            xf = x.flatten(0, -2)
            # same arithmetic as mlx_lm: product in the model dtype, accumulated into fp32
            self._store.H[self._key] = self._store.H.get(self._key, mx.array(0.0)) + xf.T @ xf
        return self.module(x, *args, **kwargs)


def input_groups(names):
    """{linear name: group key}, sharing a key only when every member of a known group exists."""
    group = {n: n for n in names}
    for members in SHARED_INPUTS:
        if all(m in group for m in members):
            for m in members:
                group[m] = members[0]
    return group


def inverse_hessian(H):
    """Upper Cholesky factor of H⁻¹ after 1% damping (identical to mlx_lm)."""
    with mx.stream(mx.cpu):
        H = mx.array(H)  # copy: the damping below writes in place, and q/k/v share one H
        damp = 1e-2 * mx.mean(mx.diag(H))
        diag = mx.arange(H.shape[0])
        H[diag, diag] += damp
        H = mx.linalg.cholesky(H)
        H = mx.linalg.cholesky_inv(H)
        return mx.linalg.cholesky(H, upper=True)


def make_error_fn(bits):
    @mx.compile
    def gptq_error(w, d, scales, biases):
        n_bins = 2**bits - 1
        q = mx.clip(mx.round((w - biases) / scales), 0.0, n_bins)
        return (w - (scales * q + biases)) / d
    return gptq_error


def gptq_layer(module, H, bits, group_size, error_fn):
    """Quantize one Linear given its Hessian. This loop is mlx_lm's, line for line:
    round one column, then push its rounding error onto the columns not yet rounded,
    weighted by H⁻¹ so the layer's output on calibration text changes as little as possible."""
    Hinv = inverse_hessian(H)
    mx.eval(Hinv)
    orig_type = module.weight.dtype
    W = module.weight.astype(mx.float32)
    all_scales, all_biases = [], []
    for i in range(0, W.shape[-1], group_size):
        j = i + group_size
        Wl = W[..., i:j]
        err = mx.zeros_like(Wl)
        _, scales, biases = mx.quantize(Wl, bits=bits, group_size=group_size)  # this group's grid
        all_scales.append(scales)
        all_biases.append(biases)
        for k in range(i, j):
            e = error_fn(W[..., k:k + 1], Hinv[k, k], scales, biases)
            W[..., k:j] -= e @ Hinv[k:k + 1, k:j]  # compensate within the group now
            err[..., k - i:k - i + 1] = e
            mx.eval(err, W)
        W[..., j:] -= err @ Hinv[i:j, j:]  # and all later groups once per group
    scales = mx.concatenate(all_scales, axis=-1)
    biases = mx.concatenate(all_biases, axis=-1)
    layer = module.to_quantized(bits=bits, group_size=group_size)
    layer.weight = pack(W, bits, scales, biases)
    layer.scales = scales
    layer.biases = biases
    layer.set_dtype(orig_type)
    mx.eval(layer)
    return layer


def _catch_and_run(block, xs, run):
    """Install Catchers on `block`'s GPTQ Linears, run it over xs; return (catchers, groups, H store, outputs)."""
    linears = [(k, l) for k, l in tree_flatten(block.leaf_modules(), is_leaf=nn.Module.is_module)
               if isinstance(l, GPTQ_TYPES)]
    groups = input_groups([k for k, _ in linears])
    store = HessianStore()
    catchers = [(k, Catcher(l, store, groups[k], leader=groups[k] == k)) for k, l in linears]
    block.update_modules(tree_unflatten(catchers))
    ys = []
    for x in xs:
        y = run(x)
        mx.eval(y, *store.H.values())
        ys.append(y)
    return catchers, groups, store, ys


def gptq_blockwise(model, data, bits, group_size, fallback_bits, fallback_group_size, batch_size=8,
                   sequential=False, log=print):
    inner = model.model
    error_fn = make_error_fn(bits)
    xs = [inner.embed_tokens(data[s:s + batch_size]) for s in range(0, len(data), batch_size)]
    mx.eval(xs)
    mask = "causal"  # what create_attention_mask returns for full sequences without a cache
    t0 = time.time()
    for b, block in enumerate(inner.layers):
        catchers, groups, store, ys = _catch_and_run(block, xs, lambda x: block(x, mask, None))
        quantized = [(k, gptq_layer(c.module, store.H[groups[k]], bits, group_size, error_fn)) for k, c in catchers]
        block.update_modules(tree_unflatten(quantized))
        store.H.clear()
        if sequential:
            ys = [block(x, mask, None) for x in xs]
            mx.eval(ys)
        xs = ys
        mx.clear_cache()
        log(f"block {b + 1}/{len(inner.layers)}  {time.time() - t0:.0f}s  "
            f"peak MLX memory {mx.get_peak_memory() / 2**30:.2f} GB")

    head = getattr(model, "lm_head", None)
    if isinstance(head, GPTQ_TYPES):  # untied output layer: GPTQ on the final normed states
        holder = nn.Module()
        holder.lm_head = head
        catchers, groups, store, _ = _catch_and_run(holder, xs, lambda x: holder.lm_head(inner.norm(x)))
        model.lm_head = gptq_layer(catchers[0][1].module, store.H["lm_head"], bits, group_size, error_fn)
        store.H.clear()

    config = {"bits": bits, "group_size": group_size}
    fallback = {"bits": fallback_bits, "group_size": fallback_group_size}
    rest = [(k, l.to_quantized(**fallback)) for k, l in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
            if hasattr(l, "to_quantized")]
    for k, _ in rest:
        config[k] = fallback
    if rest:
        model.update_modules(tree_unflatten(rest))
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local BF16 snapshot path")
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--fallback-bits", type=int, default=6)
    parser.add_argument("--fallback-group-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sequential", action="store_true",
                        help="Calibrate each block on the quantized previous blocks' outputs (GPTQ paper); not the mlx_lm default")
    args = parser.parse_args()

    mx.set_cache_limit(int(0.5 * 2**30))
    mx.random.seed(args.seed)  # same order as mlx_lm: seed, load, then draw calibration chunks
    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    data = load_data(tokenizer, args.num_samples, args.sequence_length)
    model, config["quantization"] = gptq_blockwise(model, data, args.bits, args.group_size, args.fallback_bits,
                                                   args.fallback_group_size, sequential=args.sequential)
    print(f"Quantized model with {compute_bits_per_weight(model):.3f} bits per weight; "
          f"peak MLX memory {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)
    save(args.mlx_path, args.model, model, tokenizer, config)


if __name__ == "__main__":
    main()
