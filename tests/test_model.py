import pytest
import torch

from ra_rl.features import (
    CAND_DIM,
    FEATURE_VERSION,
    GLOBAL_DIM,
    MAX_TARGETS,
    N_ARMY_ACTIONS,
    N_ITEM_ACTIONS,
    PRIV_DIM,
)
from ra_rl.model import CommanderNet
from ra_rl.vocab import TACTICAL_CHANNELS


def test_commandernet_decoupled_critic_and_multichannel_logits():
    net = CommanderNet(hidden=64, cand_hidden=32)
    B = 4
    g = torch.randn(B, GLOBAL_DIM)
    priv = torch.randn(B, PRIV_DIM)

    # Value works both with priv and with default priv=None
    v_priv = net.value(g, priv)
    v_zero = net.value(g)
    assert v_priv.shape == (B,) and v_zero.shape == (B,)

    # Value backward MUST NOT touch any actor parameter
    net.zero_grad()
    v_priv.sum().backward()
    for p in net.actor_parameters():
        assert p.grad is None
    assert any(p.grad is not None for p in net.critic_parameters())

    # Item logits
    queue = torch.randint(0, 6, (B,))
    item_mask = torch.ones(B, N_ITEM_ACTIONS, dtype=torch.bool)
    item_mask[:, 5:] = False
    logits = net.item_logits(g, queue, item_mask)
    assert logits.shape == (B, N_ITEM_ACTIONS)
    assert (logits[:, 5:] < -1e8).all()

    # Tactical logits across MAIN, HARASS, AIR, SUPPORT channels
    cand = torch.randn(B, MAX_TARGETS, CAND_DIM)
    army_mask = torch.ones(B, N_ARMY_ACTIONS, dtype=torch.bool)
    army_mask[:, 3:MAX_TARGETS] = False
    army_mask[:, MAX_TARGETS + 3:2 * MAX_TARGETS] = False
    group = torch.tensor([0, 1, 2, len(TACTICAL_CHANNELS) - 1], dtype=torch.long)
    a_logits = net.army_logits(g, cand, army_mask, group)
    assert a_logits.shape == (B, N_ARMY_ACTIONS)
    assert (a_logits[:, 3:MAX_TARGETS] < -1e8).all()
    assert (a_logits[:, MAX_TARGETS + 3:2 * MAX_TARGETS] < -1e8).all()

    # Ability logits (17 outputs: 0..15 candidates, 16 = HOLD)
    ab_mask = torch.ones(B, 17, dtype=torch.bool)
    ab_mask[:, 5:16] = False
    ab_logits = net.ability_logits(g, cand, ab_mask, torch.zeros(B, dtype=torch.long))
    assert ab_logits.shape == (B, 17)
    assert (ab_logits[:, 5:16] < -1e8).all()
    assert (ab_logits[:, 16] > -1e8).all()

    # Macro ops logits (3 outputs: repair, sell, hold)
    macro_logits = net.macro_ops_logits(g)
    assert macro_logits.shape == (B, 3)


def test_load_checkpoint_with_missing_critic_and_reject_wrong_feature_version(tmp_path):
    net = CommanderNet(hidden=32, cand_hidden=16)
    actor_only_state = {k: v for k, v in net.state_dict().items() if not k.startswith("critic.")}
    ckpt_path = tmp_path / "actor_only.pt"
    torch.save({"config": net.config, "state": actor_only_state}, ckpt_path)

    loaded = CommanderNet.load(str(ckpt_path))
    v = loaded.value(torch.randn(2, GLOBAL_DIM), torch.randn(2, PRIV_DIM))
    assert v.shape == (2,)

    bad_path = tmp_path / "old_v2.pt"
    torch.save({"config": {**net.config, "feature_version": FEATURE_VERSION - 1}, "state": actor_only_state}, bad_path)
    with pytest.raises(ValueError, match="features v"):
        CommanderNet.load(str(bad_path))
