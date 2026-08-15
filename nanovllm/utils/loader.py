import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_eagle3_weights(model: nn.Module, path: str):
    """Load an EAGLE3 draft-head checkpoint (e.g.
    thoughtworks/Llama-3.2-3B-Instruct-Eagle3) into an Eagle3DraftModel.

    Checkpoint layout (no packed/fused projections — names map 1:1):
      fc.weight, midlayer.{hidden_norm,input_layernorm,post_attention_layernorm,
        self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}.weight,
      norm.weight, lm_head.weight, d2t (I32 buffer), t2d (BOOL, skipped).

    Special cases vs load_model:
      - d2t is a registered buffer (draft->target id diffs), not a Parameter.
      - t2d is a training-only mask and is ignored at inference.
      - embed_tokens is absent (shared from the target model after loading).
    """
    loaded = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                if name == "t2d":
                    continue
                if name == "d2t":
                    model.get_buffer("d2t").copy_(f.get_tensor(name))
                    loaded.add(name)
                    continue
                param = model.get_parameter(name)
                param.data.copy_(f.get_tensor(name))
                loaded.add(name)

    # Catch silent missing-weight bugs early (an unloaded draft weight would
    # produce garbage output that is hard to attribute).
    for pname, _ in model.named_parameters():
        if pname.startswith("embed_tokens"):
            continue
        if pname not in loaded:
            raise RuntimeError(f"EAGLE3 draft weight missing in checkpoint: {pname}")
    if "d2t" not in loaded:
        raise RuntimeError("EAGLE3 draft checkpoint missing 'd2t' mapping tensor")
