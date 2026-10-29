import torch

from ra_rl.features import CAND_DIM, GLOBAL_DIM, MAX_TARGETS, N_ITEM_ACTIONS
from ra_rl.model import CommanderNet


def test_commandernet_value_and_logits():
    net = CommanderNet(hidden=64, cand_hidden=32)
    B = 4
    g = torch.randn(B, GLOBAL_DIM)

    # Value
    v = net.value(g)
    assert v.shape == (B,)

    # Item logits
    queue = torch.randint(0, 6, (B,))
    item_mask = torch.ones(B, N_ITEM_ACTIONS, dtype=torch.bool)
    item_mask[:, 5:] = False
    logits = net.item_logits(g, queue, item_mask)
    assert logits.shape == (B, N_ITEM_ACTIONS)
    assert (logits[:, 5:] < -1e8).all()

    # Army logits
    cand = torch.randn(B, MAX_TARGETS, CAND_DIM)
    army_mask = torch.ones(B, MAX_TARGETS + 2, dtype=torch.bool)
    army_mask[:, 3:MAX_TARGETS] = False
    a_logits = net.army_logits(g, cand, army_mask)
    assert a_logits.shape == (B, MAX_TARGETS + 2)
    assert (a_logits[:, 3:MAX_TARGETS] < -1e8).all()


def test_load_bc_checkpoint_with_missing_value_head():
    # Load Phase 2 BC model (has no value_head in state dict)
    net = CommanderNet.load("runs/phase2/bc_s2/model.pt")
    assert hasattr(net, "value_head")
    g = torch.randn(2, GLOBAL_DIM)
    v = net.value(g)
    assert v.shape == (2,)
