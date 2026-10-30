"""Model families and their precision variants.

BF16 checkpoints come from Hugging Face; lower precisions are built locally from the
same BF16 weights (MLX affine quantization, group size 64) so only precision changes.
"""
import json
import subprocess
import sys
from pathlib import Path

from proofwriter_test import ROOT

FAMILIES = {
    "qwen3-4b-instruct-2507": "mlx-community/Qwen3-4B-Instruct-2507-bf16",
    "qwen3-0.6b": "mlx-community/Qwen3-0.6B-bf16",
    "qwen3-1.7b": "mlx-community/Qwen3-1.7B-bf16",
    "qwen3-4b": "mlx-community/Qwen3-4B-bf16",
}
DEFAULT_FAMILY = "qwen3-4b-instruct-2507"
BITS = (8, 6, 5, 4, 3)
# "gptq4": GPTQ 4-bit (mlx_lm.quant.gptq) — same bits and group size as RTN "4bit", so only the
# rounding method differs. Non-Linear layers (the tied embedding/output layer in Qwen3) use the same
# 4-bit/64 RTN as the "4bit" model instead of mlx_lm's 6-bit default.
PRECISION_NAMES = ["bf16"] + [f"{b}bit" for b in BITS] + ["gptq4"]
GROUP_SIZE = 64
GPTQ = {"bits": 4, "group_size": GROUP_SIZE, "fallback_bits": 4, "fallback_group_size": GROUP_SIZE,
        "num_samples": -1, "sequence_length": 512, "seed": 123}
# mlx_lm's GPTQ holds one fp32 Hessian (d_in x d_in) per Linear layer for the whole model at once.
GPTQ_HESSIAN_BUDGET_GB = 8.0
# Hybrid Qwen3 checkpoints answer directly only with thinking disabled; templates
# without the switch (e.g. Instruct-2507) ignore it.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def model_path(family, precision):
    repo = FAMILIES[family]
    if precision == "bf16":
        return repo
    name = repo.split("/")[-1].removesuffix("-bf16")
    return str(ROOT / "models" / f"{name}-{precision}")


def gptq_hessian_gb(config):
    """Memory for mlx_lm GPTQ's Hessians: per layer, q/k/v/gate/up see d_model inputs,
    o_proj sees n_heads*head_dim, down_proj sees d_ff; each Hessian is d_in^2 fp32 values."""
    d, ff = config["hidden_size"], config["intermediate_size"]
    attn = config["num_attention_heads"] * config.get("head_dim", d // config["num_attention_heads"])
    per_layer = 5 * d * d + attn * attn + ff * ff
    return per_layer * 4 * config["num_hidden_layers"] / 2**30


def ensure_quantized(family, precision):
    """Build a quantized checkpoint from the cached BF16 snapshot if it is missing."""
    path = Path(model_path(family, precision))
    if precision == "bf16" or path.exists():
        return
    from huggingface_hub import snapshot_download
    source = snapshot_download(FAMILIES[family], local_files_only=True)
    if precision == "gptq4":
        need = gptq_hessian_gb(json.loads((Path(source) / "config.json").read_text()))
        if need > GPTQ_HESSIAN_BUDGET_GB:
            raise MemoryError(f"GPTQ for {family} needs ~{need:.1f} GB of Hessians (budget "
                              f"{GPTQ_HESSIAN_BUDGET_GB} GB); not safe on this machine")
        args = [f"--{k.replace('_', '-')}={v}" for k, v in GPTQ.items()]
        subprocess.run([sys.executable, "-m", "mlx_lm.quant.gptq", "--model", source, "--mlx-path", str(path), *args],
                       check=True)
        return
    bits = precision.removesuffix("bit")
    subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", source, "--mlx-path", str(path),
                    "-q", "--q-bits", bits, "--q-group-size", str(GROUP_SIZE)], check=True)
