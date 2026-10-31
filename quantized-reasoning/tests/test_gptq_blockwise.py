from pathlib import Path
import sys
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gptq_blockwise  # noqa: E402


def tiny_qwen3(tied=True, seed=0):
    """A 2-block Qwen3 small enough to quantize in a second (dims are multiples of group size 64)."""
    from mlx_lm.models.qwen3 import Model, ModelArgs
    args = ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
                     num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=1,
                     max_position_embeddings=64, head_dim=32, tie_word_embeddings=tied, rope_theta=10000.0)
    mx.random.seed(seed)
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def clone(model, tied):
    other = tiny_qwen3(tied, seed=99)
    other.load_weights(tree_flatten(model.parameters()))
    mx.eval(other.parameters())
    return other


def quantized_params(model):
    return {k: v for k, v in tree_flatten(model.parameters()) if k.endswith(("weight", "scales", "biases"))}


class BlockwiseGPTQTests(unittest.TestCase):
    def compare_to_mlx_lm(self, tied):
        from mlx_lm.quant.gptq import gptq_quantize
        base = tiny_qwen3(tied)
        data = mx.random.randint(0, 128, (10, 16), key=mx.random.key(1))  # 10 samples: last batch is partial
        ref, ref_cfg = gptq_quantize(clone(base, tied), data, 4, 64, 4, 64)
        ours, our_cfg = gptq_blockwise.gptq_blockwise(clone(base, tied), data, 4, 64, 4, 64, log=lambda _: None)
        self.assertEqual(ref_cfg, our_cfg)
        a, b = quantized_params(ref), quantized_params(ours)
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            self.assertEqual(a[k].dtype, b[k].dtype, k)
            self.assertTrue(mx.array_equal(a[k], b[k]).item(), f"{k} differs from mlx_lm")

    def test_matches_mlx_lm_tied_embeddings(self):
        self.compare_to_mlx_lm(tied=True)

    def test_matches_mlx_lm_untied_lm_head(self):
        self.compare_to_mlx_lm(tied=False)

    def test_input_groups_share_hessians(self):
        names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        groups = gptq_blockwise.input_groups(names)
        self.assertEqual(len(set(groups.values())), 4)  # qkv, o, gate/up, down
        self.assertEqual(gptq_blockwise.input_groups(["self_attn.q_proj"]), {"self_attn.q_proj": "self_attn.q_proj"})

    def test_sequential_mode_runs_and_differs(self):
        base = tiny_qwen3()
        data = mx.random.randint(0, 128, (8, 16), key=mx.random.key(2))
        plain, _ = gptq_blockwise.gptq_blockwise(clone(base, True), data, 4, 64, 4, 64, log=lambda _: None)
        seq, _ = gptq_blockwise.gptq_blockwise(clone(base, True), data, 4, 64, 4, 64, sequential=True,
                                               log=lambda _: None)
        a, b = quantized_params(plain), quantized_params(seq)
        first = "model.layers.0.self_attn.q_proj.weight"
        self.assertTrue(mx.array_equal(a[first], b[first]).item())  # block 0 sees the same inputs
        self.assertFalse(all(mx.array_equal(a[k], b[k]).item() for k in a if ".layers.1." in k))

    def test_gptq_beats_round_to_nearest_on_correlated_inputs(self):
        """The point of GPTQ: lower layer-output error than rounding each weight on its own."""
        mx.random.seed(3)
        layer = nn.Linear(128, 32, bias=False)
        base = mx.random.normal((512, 8))
        x = base @ mx.random.normal((8, 128)) + 0.05 * mx.random.normal((512, 128))  # strongly correlated
        H = mx.array(0.0) + x.T @ x
        gq = gptq_blockwise.gptq_layer(layer, H, 4, 64, gptq_blockwise.make_error_fn(4))
        rtn = layer.to_quantized(bits=4, group_size=64)
        target = layer(x)
        err = lambda q: mx.mean((q(x) - target) ** 2).item()
        self.assertLess(err(gq), 0.7 * err(rtn))


if __name__ == "__main__":
    unittest.main()
