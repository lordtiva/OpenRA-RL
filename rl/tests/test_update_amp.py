# -*- coding: utf-8 -*-
"""BPTT batcheado + prefetch GPU: no cambia el grafo, no pinea el elite."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from rl.imitation import EliteBuffer
from rl.network import (
    HIDDEN_DIM, N_ACTION_TYPES, TYPE_TO_IDX, _heads_used, _mask_illegal,
    AlphaLiteNet,
)
from rl.obs_encoding import SCALAR_DIM, UNIT_FEAT_DIM
from rl.trainer import PPOTrainer, prefetch_steps

ok = True


def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    ok = ok and bool(cond)


def _dummy_step(ep=0, kind="no_op", H=8, W=8, U=4, V=8):
    t_idx = TYPE_TO_IDX[kind]
    spatial = torch.zeros(1, 9, H, W)
    batch = {
        "spatial": spatial,
        "scalars": torch.zeros(1, SCALAR_DIM),
        "unit_feats": torch.zeros(1, U, UNIT_FEAT_DIM),
        "unit_valid": torch.zeros(1, U, dtype=torch.bool),
        "type_mask": torch.ones(1, N_ACTION_TYPES, dtype=torch.bool),
        "cell_mask": torch.ones(1, H * W, dtype=torch.bool),
        "item_indices": torch.zeros(1, V, dtype=torch.long),
        "item_mask": torch.zeros(1, V, dtype=torch.bool),
        "train_slot_mask": torch.zeros(1, V, dtype=torch.bool),
        "build_slot_mask": torch.zeros(1, V, dtype=torch.bool),
        "unit_role_ids": torch.zeros(1, U, dtype=torch.long),
        "unit_own_mask": torch.zeros(1, U, dtype=torch.bool),
    }
    batch["unit_valid"][0, 0] = True
    batch["unit_own_mask"][0, 0] = True
    return {
        "_ep": ep,
        "batch": batch,
        "action": {
            "type": torch.tensor([t_idx]),
            "unit_slot": torch.tensor([0]),
            "cell_flat": torch.tensor([0]),
            "item_slot": torch.tensor([0]),
            "had_item": torch.tensor([False]),
            "log_prob": torch.tensor([0.0]),
        },
        "h_in": torch.zeros(1, HIDDEN_DIM),
        "adv": 0.1,
        "ret": 0.0,
        "value_pred": 0.0,
    }


print("=== update amp / batched BPTT ===")

x16 = torch.zeros(2, 4, dtype=torch.float16)
ill = torch.tensor([[True, False, False, True],
                    [False, False, True, False]])
y16 = _mask_illegal(x16, ill)
check("half mask no overflow", y16.dtype == torch.float16)
check("half illegal ≈ -1e4", float(y16[0, 0]) < -1000)
check("half legal intacto", float(y16[0, 1]) == 0.0)
x32 = torch.zeros(1, 3)
y32 = _mask_illegal(x32, torch.tensor([[True, False, False]]))
check("fp32 illegal sigue -1e9", float(y32[0, 0]) < -1e8)

if torch.cuda.is_available():
    net_c = AlphaLiteNet().cuda().eval()
    h_c = torch.zeros(1, HIDDEN_DIM, device="cuda")
    m_c = torch.ones(1, N_ACTION_TYPES, dtype=torch.bool, device="cuda")
    m_c[0, 3] = False
    try:
        amp_ctx = torch.amp.autocast("cuda")
    except TypeError:
        amp_ctx = torch.cuda.amp.autocast()
    with amp_ctx:
        logits_c = net_c._logits_type(h_c, m_c)
    check("amp cuda _logits_type no crashea",
          torch.isfinite(logits_c.float()).any().item())
    del net_c, h_c, m_c
    torch.cuda.empty_cache()
else:
    check("amp cuda _logits_type skipped (no GPU)", True)

t = torch.tensor([
    TYPE_TO_IDX["no_op"], TYPE_TO_IDX["train"],
    TYPE_TO_IDX["army_attack_move"], TYPE_TO_IDX["move"],
])
u, c, i = _heads_used(t)
check("heads no_op: ninguna extra",
      (not bool(u[0])) and (not bool(c[0])) and (not bool(i[0])))
check("heads train: solo item",
      (not bool(u[1])) and (not bool(c[1])) and bool(i[1]))
check("heads army_attack_move: solo celda",
      (not bool(u[2])) and bool(c[2]) and (not bool(i[2])))
check("heads move: unidad+celda",
      bool(u[3]) and bool(c[3]) and (not bool(i[3])))

torch.manual_seed(0)
net = AlphaLiteNet()
net.eval()
seg_a = [_dummy_step(ep=0, kind="no_op"), _dummy_step(ep=0, kind="train")]
seg_b = [_dummy_step(ep=1, kind="move")]
with torch.no_grad():
    lp1, e1, v1 = net.evaluate_actions_seq(seg_a, "cpu")
    lp2, e2, v2 = net.evaluate_actions_seq(seg_b, "cpu")
    lpb, eb, vb, valid = net.evaluate_actions_seq_batch([seg_a, seg_b], "cpu")
check("batch T=max(2,1)", lpb.shape == (2, 2) and valid[0].sum() == 2
      and valid[1].sum() == 1)
check("batch vs seq seg A lp", torch.allclose(lpb[0, :2], lp1, atol=1e-5))
check("batch vs seq seg B lp", torch.allclose(lpb[1, :1], lp2, atol=1e-5))
check("batch pad B no cuenta", bool(valid[1, 1] == False))

# grafo: un backward batcheado llega a los pesos
net.train()
lpb, eb, vb, valid = net.evaluate_actions_seq_batch([seg_a, seg_b], "cpu")
loss = (lpb * valid.float()).sum()
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0).item()
check("batch BPTT produce grad finito", math.isfinite(gn) and gn > 0)

cpu_h = torch.zeros(1, HIDDEN_DIM)
step = _dummy_step()
step["h_in"] = cpu_h
elite = EliteBuffer(cap_steps=10)
elite.add_episode([step], {"result": "win", "ticks": 1000})
copied = prefetch_steps(list(elite._episodes[0]["steps"]), "cpu", inplace=False)
check("prefetch copy no comparte dict", copied[0] is not elite._episodes[0]["steps"][0])
check("elite h_in sigue CPU", elite._episodes[0]["steps"][0]["h_in"].device.type == "cpu")

tr = PPOTrainer(AlphaLiteNet(), lr=1e-4, device="cpu")
st = tr.update(
    [_dummy_step(ep=0, kind="no_op"), _dummy_step(ep=0, kind="train"),
     _dummy_step(ep=1, kind="move")],
    epochs=1, batch_size=64)
check("ppo batched update finito", math.isfinite(st.get("pi_loss", 0)))
check("ppo batched n samples", st.get("n") == 3)

print("=== fin update amp ===")
if not ok:
    sys.exit(1)
print("OK")
