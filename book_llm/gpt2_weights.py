import importlib.util
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .model import GPT_CONFIG_124M, GPTModel


GPT_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)

GPT2_MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def build_gpt2_config(model_name="gpt2-small (124M)", base_config=None):
    cfg = (base_config or GPT_CONFIG_124M).copy()
    cfg.update(GPT2_MODEL_CONFIGS[model_name])
    cfg.update({"context_length": 1024, "qkv_bias": True})
    return cfg


def model_name_to_size(model_name):
    return model_name.split()[-1].lstrip("(").rstrip(")")


def ensure_gpt_download_helper(
    destination="gpt_download.py",
    url=GPT_DOWNLOAD_URL,
):
    destination = Path(destination)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    return destination


def import_download_and_load_gpt2(helper_path="gpt_download.py"):
    helper_path = Path(helper_path)
    if not helper_path.exists():
        helper_path = ensure_gpt_download_helper(helper_path)

    module_name = "gpt_download"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import GPT-2 helper from {helper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.download_and_load_gpt2


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        block = params["blocks"][b]
        q_w, k_w, v_w = np.split(block["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight,
            q_w.T,
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight,
            k_w.T,
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight,
            v_w.T,
        )

        q_b, k_b, v_b = np.split(block["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias,
            q_b,
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias,
            k_b,
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias,
            v_b,
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            block["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            block["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            block["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            block["mlp"]["c_fc"]["b"],
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            block["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            block["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            block["ln_1"]["g"],
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            block["ln_1"]["b"],
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            block["ln_2"]["g"],
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            block["ln_2"]["b"],
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def load_pretrained_gpt2(model_name="gpt2-small (124M)", models_dir="gpt2"):
    cfg = build_gpt2_config(model_name)
    download_and_load_gpt2 = import_download_and_load_gpt2()
    settings, params = download_and_load_gpt2(
        model_size=model_name_to_size(model_name),
        models_dir=models_dir,
    )
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    model.eval()
    return model, cfg, settings
