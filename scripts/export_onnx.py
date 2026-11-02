"""export_onnx.py — Export CommanderNet PyTorch checkpoint to ONNX models.

Exports:
- commander_item.onnx: (g, queue, mask) -> item_logits
- commander_army.onnx: (g, cand, mask, group) -> army_logits
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
        self.cand_enc = net.cand_enc
        self.cand_score = net.cand_score
        self.army_fixed = net.army_fixed

    def forward(self, g: torch.Tensor, cand: torch.Tensor, mask: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        h = self.trunk(g) + self.tactical_emb(group)
        c = self.cand_enc(cand)
        hc = torch.cat([h.unsqueeze(1).expand(-1, MAX_TARGETS, -1), c], dim=-1)
        pair_scores = self.cand_score(hc)
        logits = torch.cat(
            [pair_scores[..., 0], pair_scores[..., 1], self.army_fixed(h)],
            dim=-1,
        )
        return torch.where(mask, logits, torch.full_like(logits, -1e9))


def export_models(ckpt_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    net = CommanderNet.load(ckpt_path, device="cpu").eval()

    item_model = ItemModel(net).eval()
    army_model = ArmyModel(net).eval()

    # 1. Export Item Model
    item_onnx_path = out_dir / "commander_item.onnx"
    dummy_g = torch.randn(1, GLOBAL_DIM, dtype=torch.float32)
    dummy_queue = torch.tensor([0], dtype=torch.int64)
    dummy_item_mask = torch.ones(1, N_ITEM_ACTIONS, dtype=torch.bool)

    torch.onnx.export(
        item_model,
        (dummy_g, dummy_queue, dummy_item_mask),
        str(item_onnx_path),
        input_names=["g", "queue", "mask"],
        output_names=["logits"],
        dynamic_axes={
            "g": {0: "batch"},
            "queue": {0: "batch"},
            "mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported item model: {item_onnx_path} ({item_onnx_path.stat().st_size // 1024} KB)")

    # 2. Export Army Model
    army_onnx_path = out_dir / "commander_army.onnx"
    dummy_cand = torch.randn(1, MAX_TARGETS, CAND_DIM, dtype=torch.float32)
    dummy_army_mask = torch.ones(1, N_ARMY_ACTIONS, dtype=torch.bool)
    dummy_group = torch.tensor([0], dtype=torch.int64)

    torch.onnx.export(
        army_model,
        (dummy_g, dummy_cand, dummy_army_mask, dummy_group),
        str(army_onnx_path),
        input_names=["g", "cand", "mask", "group"],
        output_names=["logits"],
        dynamic_axes={
            "g": {0: "batch"},
            "cand": {0: "batch"},
            "mask": {0: "batch"},
            "group": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported army model: {army_onnx_path} ({army_onnx_path.stat().st_size // 1024} KB)")

    # 3. Verify numerical consistency with onnxruntime
    print("\nVerifying numerical outputs between PyTorch and ONNX Runtime...")
    with torch.no_grad():
        pt_item_out = item_model(dummy_g, dummy_queue, dummy_item_mask).numpy()
        pt_army_out = army_model(dummy_g, dummy_cand, dummy_army_mask, dummy_group).numpy()

    sess_item = ort.InferenceSession(str(item_onnx_path))
    ort_item_out = sess_item.run(
        None,
        {
            "g": dummy_g.numpy(),
            "queue": dummy_queue.numpy(),
            "mask": dummy_item_mask.numpy(),
        },
    )[0]

    item_diff = np.max(np.abs(pt_item_out - ort_item_out))
    print(f"Item model max absolute difference: {item_diff:.2e}")
    assert item_diff < 1e-4, f"Item model discrepancy too large: {item_diff}"

    sess_army = ort.InferenceSession(str(army_onnx_path))
    ort_army_out = sess_army.run(
        None,
        {
            "g": dummy_g.numpy(),
            "cand": dummy_cand.numpy(),
            "mask": dummy_army_mask.numpy(),
            "group": dummy_group.numpy(),
        },
    )[0]

    army_diff = np.max(np.abs(pt_army_out - ort_army_out))
    print(f"Army model max absolute difference: {army_diff:.2e}")
    assert army_diff < 1e-4, f"Army model discrepancy too large: {army_diff}"

    print("\n[SUCCESS] All ONNX models exported and verified mathematically identical!")
    return item_onnx_path, army_onnx_path


if __name__ == "__main__":
    ckpt = ROOT_DIR / "runs" / "v4" / "selfplay" / "checkpoint_best.pt"
    out = ROOT_DIR / "onnx_models"
    export_models(str(ckpt), str(out))
