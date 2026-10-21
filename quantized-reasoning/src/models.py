"""Model families and their precision variants.

BF16 checkpoints come from Hugging Face; lower precisions are built locally from the
same BF16 weights (MLX affine quantization, group size 64) so only precision changes.
"""
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
PRECISION_NAMES = ["bf16"] + [f"{b}bit" for b in BITS]
GROUP_SIZE = 64
# Hybrid Qwen3 checkpoints answer directly only with thinking disabled; templates
# without the switch (e.g. Instruct-2507) ignore it.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def model_path(family, precision):
    repo = FAMILIES[family]
    if precision == "bf16":
        return repo
    name = repo.split("/")[-1].removesuffix("-bf16")
    return str(ROOT / "models" / f"{name}-{precision}")


def ensure_quantized(family, precision):
    """Build a quantized checkpoint from the cached BF16 snapshot if it is missing."""
    path = Path(model_path(family, precision))
    if precision == "bf16" or path.exists():
        return
    from huggingface_hub import snapshot_download
    source = snapshot_download(FAMILIES[family], local_files_only=True)
    bits = precision.removesuffix("bit")
    subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", source, "--mlx-path", str(path),
                    "-q", "--q-bits", bits, "--q-group-size", str(GROUP_SIZE)], check=True)
