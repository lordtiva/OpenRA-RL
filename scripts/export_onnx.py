"""export_onnx.py — Export CommanderNet v5 PyTorch checkpoint to ONNX models.

Exports:
- commander_item.onnx: (g, queue, mask) -> item_logits
- commander_army.onnx: (g, cand, mask, group) -> army_logits
- commander_ability.onnx: (g, cand, mask, ability_id) -> ability_logits
- commander_macro.onnx: (g, mask) -> macro_ops_logits
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ra_rl.features import CAND_DIM, GLOBAL_DIM, MAX_TARGETS, N_ARMY_ACTIONS, N_ITEM_ACTIONS
from ra_rl.model import CommanderNet


class ItemModel(nn.Module):
    def __init__(self, net: CommanderNet):
        super().__init__()
        self.trunk = net.trunk
        self.queue_emb = net.queue_emb
        self.item_head = net.item_head

    def forward(self, g: torch.Tensor, queue: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g) + self.queue_emb(queue)
        logits = self.item_head(h)
        return torch.where(mask, logits, torch.full_like(logits, -1e9))


class ArmyModel(nn.Module):
    def __init__(self, net: CommanderNet):
        super().__init__()
        self.trunk = net.trunk
        self.tactical_emb = net.tactical_emb
        self.cls_token_proj = net.cls_token_proj
        self.cand_proj = net.cand_proj
        self.spatial_transformer = net.spatial_transformer
        self.cand_score = net.cand_score
        self.army_fixed = net.army_fixed

    def forward(self, g: torch.Tensor, cand: torch.Tensor, mask: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g) + self.tactical_emb(group)
        cls_tok = self.cls_token_proj(h).unsqueeze(1)
        cand_toks = self.cand_proj(cand)
        tokens = torch.cat([cand_toks, cls_tok], dim=1)
        trans_tokens = self.spatial_transformer(tokens)
        cand_ctx = trans_tokens[:, :MAX_TARGETS, :]
        pair_scores = self.cand_score(cand_ctx)
        logits = torch.cat(
            [pair_scores[..., 0], pair_scores[..., 1], self.army_fixed(h)],
            dim=-1,
        )
        return torch.where(mask, logits, torch.full_like(logits, -1e9))


class AbilityModel(nn.Module):
    def __init__(self, net: CommanderNet):
        super().__init__()
        self.trunk = net.trunk
        self.ability_emb = net.ability_emb
        self.cls_token_proj = net.cls_token_proj
        self.cand_proj = net.cand_proj
        self.spatial_transformer = net.spatial_transformer
        self.ability_head = net.ability_head

    def forward(self, g: torch.Tensor, cand: torch.Tensor, mask: torch.Tensor, ability_id: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g) + self.ability_emb(ability_id)
        cls_tok = self.cls_token_proj(h).unsqueeze(1)
        cand_toks = self.cand_proj(cand)
        tokens = torch.cat([cand_toks, cls_tok], dim=1)
        trans_tokens = self.spatial_transformer(tokens)
        logits = self.ability_head(trans_tokens).squeeze(-1)
        return torch.where(mask, logits, torch.full_like(logits, -1e9))


class MacroOpsModel(nn.Module):
    def __init__(self, net: CommanderNet):
        super().__init__()
        self.trunk = net.trunk
        self.macro_ops_head = net.macro_ops_head

    def forward(self, g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g)
        logits = self.macro_ops_head(h)
        return torch.where(mask, logits, torch.full_like(logits, -1e9))


def export_models(ckpt_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    net = CommanderNet.load(ckpt_path, device="cpu").eval()

    item_model = ItemModel(net).eval()
    army_model = ArmyModel(net).eval()
    ability_model = AbilityModel(net).eval()
    macro_model = MacroOpsModel(net).eval()

    dummy_g = torch.randn(1, GLOBAL_DIM, dtype=torch.float32)
    dummy_cand = torch.randn(1, MAX_TARGETS, CAND_DIM, dtype=torch.float32)

    # 1. Export Item Model
    item_onnx_path = out_dir / "commander_item.onnx"
    dummy_queue = torch.tensor([0], dtype=torch.int64)
    dummy_item_mask = torch.ones(1, N_ITEM_ACTIONS, dtype=torch.bool)
    torch.onnx.export(
        item_model,
        (dummy_g, dummy_queue, dummy_item_mask),
        str(item_onnx_path),
        input_names=["g", "queue", "mask"],
        output_names=["logits"],
        dynamic_axes={"g": {0: "batch"}, "queue": {0: "batch"}, "mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported item model: {item_onnx_path} ({item_onnx_path.stat().st_size // 1024} KB)")

    # 2. Export Army Model
    army_onnx_path = out_dir / "commander_army.onnx"
    dummy_army_mask = torch.ones(1, N_ARMY_ACTIONS, dtype=torch.bool)
    dummy_group = torch.tensor([0], dtype=torch.int64)
    torch.onnx.export(
        army_model,
        (dummy_g, dummy_cand, dummy_army_mask, dummy_group),
        str(army_onnx_path),
        input_names=["g", "cand", "mask", "group"],
        output_names=["logits"],
        dynamic_axes={"g": {0: "batch"}, "cand": {0: "batch"}, "mask": {0: "batch"}, "group": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported army model: {army_onnx_path} ({army_onnx_path.stat().st_size // 1024} KB)")

    # 3. Export Ability Model
    ability_onnx_path = out_dir / "commander_ability.onnx"
    dummy_ab_mask = torch.ones(1, 17, dtype=torch.bool)
    dummy_ab_id = torch.tensor([0], dtype=torch.int64)
    torch.onnx.export(
        ability_model,
        (dummy_g, dummy_cand, dummy_ab_mask, dummy_ab_id),
        str(ability_onnx_path),
        input_names=["g", "cand", "mask", "ability_id"],
        output_names=["logits"],
        dynamic_axes={"g": {0: "batch"}, "cand": {0: "batch"}, "mask": {0: "batch"}, "ability_id": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ability model: {ability_onnx_path} ({ability_onnx_path.stat().st_size // 1024} KB)")

    # 4. Export MacroOps Model
    macro_onnx_path = out_dir / "commander_macro.onnx"
    dummy_macro_mask = torch.ones(1, 3, dtype=torch.bool)
    torch.onnx.export(
        macro_model,
        (dummy_g, dummy_macro_mask),
        str(macro_onnx_path),
        input_names=["g", "mask"],
        output_names=["logits"],
        dynamic_axes={"g": {0: "batch"}, "mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported macro ops model: {macro_onnx_path} ({macro_onnx_path.stat().st_size // 1024} KB)")

    # 5. Verify numerical consistency with onnxruntime
    print("\nVerifying numerical outputs between PyTorch and ONNX Runtime...")
    with torch.no_grad():
        pt_item_out = item_model(dummy_g, dummy_queue, dummy_item_mask).numpy()
        pt_army_out = army_model(dummy_g, dummy_cand, dummy_army_mask, dummy_group).numpy()
        pt_ab_out = ability_model(dummy_g, dummy_cand, dummy_ab_mask, dummy_ab_id).numpy()
        pt_mo_out = macro_model(dummy_g, dummy_macro_mask).numpy()

    sess_item = ort.InferenceSession(str(item_onnx_path))
    ort_item_out = sess_item.run(None, {"g": dummy_g.numpy(), "queue": dummy_queue.numpy(), "mask": dummy_item_mask.numpy()})[0]
    assert np.max(np.abs(pt_item_out - ort_item_out)) < 1e-4

    sess_army = ort.InferenceSession(str(army_onnx_path))
    ort_army_out = sess_army.run(None, {"g": dummy_g.numpy(), "cand": dummy_cand.numpy(), "mask": dummy_army_mask.numpy(), "group": dummy_group.numpy()})[0]
    assert np.max(np.abs(pt_army_out - ort_army_out)) < 1e-4

    sess_ab = ort.InferenceSession(str(ability_onnx_path))
    ort_ab_out = sess_ab.run(None, {"g": dummy_g.numpy(), "cand": dummy_cand.numpy(), "mask": dummy_ab_mask.numpy(), "ability_id": dummy_ab_id.numpy()})[0]
    assert np.max(np.abs(pt_ab_out - ort_ab_out)) < 1e-4

    sess_mo = ort.InferenceSession(str(macro_onnx_path))
    ort_mo_out = sess_mo.run(None, {"g": dummy_g.numpy(), "mask": dummy_macro_mask.numpy()})[0]
    assert np.max(np.abs(pt_mo_out - ort_mo_out)) < 1e-4

    print("\n[SUCCESS] All 4 ONNX models exported and verified mathematically identical!")
    return item_onnx_path, army_onnx_path, ability_onnx_path, macro_onnx_path
