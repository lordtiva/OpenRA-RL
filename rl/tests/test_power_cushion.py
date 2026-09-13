# -*- coding: utf-8 -*-
"""Low-power plant: adapter gate + scalar, no scratch."""
from types import SimpleNamespace as NS

from openra_env.models import ActionType
from rl.action_adapter import (
    ActionIndex,
    Vocab,
    index_to_command_effective,
    needs_power_build,
    power_in_deficit,
    spendable_resources,
)
from rl.network import TYPE_TO_IDX, adapt_scalar_state_dict, AlphaLiteNet
from rl.obs_encoding import SCALAR_DIM, scalar_features
import torch


class _B:
    def __init__(self, typ="proc", x=5, y=5, aid=99):
        self.type = typ
        self.cell_x = x
        self.cell_y = y
        self.actor_id = aid
        self.hp_percent = 1.0
        self.is_repairing = False
        self.is_powered = True
        self.rally_x = -1
        self.rally_y = -1


class _U:
    def __init__(self, typ="e1", aid=1, x=8, y=8):
        self.type = typ
        self.actor_id = aid
        self.cell_x = x
        self.cell_y = y
        self.hp_percent = 1.0
        self.can_attack = typ not in ("harv", "mcv")
        self.is_idle = True
        self.speed = 50
        self.attack_range = 0
        self.experience_level = 0
        self.stance = 0
        self.facing = 0


def _obs(*, cash=0, ore=12000, provided=100, drained=460,
         bldgs=("fact", "proc", "powr", "tent"),
         avail=("e1", "powr", "pbox", "weap", "proc"),
         units=None, prod=()):
    if units is None:
        units = [_U("harv", 1), _U("e1", 2)]
    return NS(
        tick=36000,
        map_info=NS(height=32, width=32, map_name="fase2_a_short.oramap"),
        economy=NS(cash=cash, ore=ore, harvester_count=1,
                   power_provided=provided, power_drained=drained,
                   resource_capacity=13000),
        military=NS(kills_cost=0, deaths_cost=0, assets_value=2000,
                    units_killed=0, units_lost=0, army_value=0,
                    active_unit_count=0),
        buildings=[_B(t, aid=100 + i) for i, t in enumerate(bldgs)],
        units=list(units),
        production=list(prod),
        available_production=list(avail),
        visible_enemies=[],
        visible_enemy_buildings=[],
        spatial_map="",
        spatial_channels=9,
    )


def _aidx(obs):
    v = Vocab()
    v.seed_roles()
    return ActionIndex(obs, v)


def test_spendable_includes_silo_ore():
    obs = _obs(cash=0, ore=12983)
    assert spendable_resources(obs) == 12983
    assert power_in_deficit(obs) is True
    assert needs_power_build(obs) is True


def test_scalar_power_balance_past_2x_clamp():
    assert SCALAR_DIM == 34
    obs = _obs(provided=100, drained=570)
    sc = scalar_features(obs)
    assert sc.shape == (34,)
    # index 3 saturates at 2x → 1.0; signed col still at floor -1
    assert float(sc[3]) == 1.0
    assert float(sc[4]) == 1.0
    assert float(sc[33]) == -1.0
    ok = _obs(provided=100, drained=50)
    sc_ok = scalar_features(ok)
    assert abs(float(sc_ok[33]) - 0.125) < 1e-5


def test_adapter_masks_e1_and_pbox_when_low_power():
    obs = _obs()
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["train"]]) is False
    assert "power" in aidx.build_items
    pslot = len(aidx.train_items) + aidx.build_items.index("power")
    assert bool(aidx.build_slot_mask[pslot]) is True
    if "defense_gun" in aidx.build_items:
        dslot = len(aidx.train_items) + aidx.build_items.index("defense_gun")
        assert bool(aidx.build_slot_mask[dslot]) is False
    if "infantry_basic" in aidx.train_items:
        tslot = aidx.train_items.index("infantry_basic")
        assert bool(aidx.train_slot_mask[tslot]) is False
    assert bool(aidx.type_mask[TYPE_TO_IDX["build"]]) is True


def test_remap_train_e1_to_build_power():
    obs = _obs()
    aidx = _aidx(obs)
    t = TYPE_TO_IDX["train"]
    slot = aidx.items.index("infantry_basic") if "infantry_basic" in aidx.items else 0
    action, eff = index_to_command_effective(obs, t, 0, 0, slot, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.BUILD
    assert str(cmd.item_type).lower() in ("powr", "power")
    assert eff[0] == TYPE_TO_IDX["build"]


def test_surplus_power_does_not_mask_train():
    obs = _obs(provided=200, drained=50, cash=5000, ore=0)
    assert power_in_deficit(obs) is False
    aidx = _aidx(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["train"]]) is True
    if "infantry_basic" in aidx.train_items:
        tslot = aidx.train_items.index("infantry_basic")
        assert bool(aidx.train_slot_mask[tslot]) is True


def test_queue_busy_holds_train_does_not_overwrite():
    prod = [NS(queue_type="Building", item="proc", progress=0.4, paused=False)]
    obs = _obs(prod=prod)
    assert needs_power_build(obs) is False
    aidx = _aidx(obs)
    if "infantry_basic" in aidx.train_items:
        tslot = aidx.train_items.index("infantry_basic")
        assert bool(aidx.train_slot_mask[tslot]) is False


def test_net2net_pad_33_loads():
    net = AlphaLiteNet()
    key = "scalar_mlp.0.weight"
    raw = {k: v.clone() for k, v in net.state_dict().items()}
    raw[key] = raw[key][:, :33].contiguous()
    adapted = adapt_scalar_state_dict(net, raw)
    net2 = AlphaLiteNet()
    net2.load_state_dict(adapted, strict=False)
    assert net2.scalar_mlp[0].weight.shape[1] == 34
    assert torch.allclose(net2.scalar_mlp[0].weight[:, 33:],
                          torch.zeros(net2.scalar_mlp[0].weight.size(0), 1))


def test_adapted_shapes_changed_scalar_33_to_34():
    """Helper sees soft-pad grow even when missing/unexpected would be empty."""
    from rl.trainer import _adapted_shapes_changed, _first_adapted_shape_change

    net = AlphaLiteNet()
    key = "scalar_mlp.0.weight"
    raw = {k: v.clone() for k, v in net.state_dict().items()}
    raw[key] = raw[key][:, :33].contiguous()
    adapted = adapt_scalar_state_dict(net, raw)
    assert _adapted_shapes_changed(raw, adapted) is True
    chg = _first_adapted_shape_change(raw, adapted)
    assert chg is not None
    assert chg[0] == key
    assert chg[1] == (256, 33)
    assert chg[2] == (256, 34)
    # identical shapes → False
    same = {k: v.clone() for k, v in net.state_dict().items()}
    assert _adapted_shapes_changed(same, same) is False


def test_load_checkpoint_adam_reset_on_scalar_pad(tmp_path):
    """33->34 soft-pad must force Adam fresco (do not load mismatched opt)."""
    import os
    from rl.trainer import load_checkpoint, save_checkpoint

    net = AlphaLiteNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # touch Adam state so there is something to (not) load
    for p in net.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
            break
    opt.step()
    path = os.path.join(str(tmp_path), "ckpt.pt")
    save_checkpoint(path, net, opt, iteration=7)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    key = "scalar_mlp.0.weight"
    ckpt["net"][key] = ckpt["net"][key][:, :33].contiguous()
    torch.save(ckpt, path)

    net2 = AlphaLiteNet()
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    # fresh Adam has empty state; after successful load_state_dict it would
    # gain param groups with exp_avg matching old shapes. We assert reset:
    # either opt state stays empty-ish for the padded param, or load refused.
    before_state_keys = set(opt2.state_dict().get("state", {}).keys())
    load_checkpoint(path, net2, opt2)
    after = opt2.state_dict()
    # With do_reset=True we never call opt.load_state_dict, so state stays
    # at fresh init (empty state dict).
    assert after.get("state", {}) == {} or set(after.get("state", {}).keys()) == before_state_keys
    assert net2.scalar_mlp[0].weight.shape[1] == 34
