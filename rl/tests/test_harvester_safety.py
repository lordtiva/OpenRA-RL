# -*- coding: utf-8 -*-
"""harvesters_move is eco/safety: not K=2 push, home-ore cell, no harv combat slot."""
import numpy as np
import torch

from openra_env.models import ActionType
from rl.action_adapter import (
    ActionIndex, Vocab, TYPE_TO_IDX, harvesters_need_move,
    index_to_command, index_to_command_effective, remap_harvester_cell,
)
from rl.network import (
    ACTION_TYPES, COMBAT_PUSH_TYPES, HIDDEN_DIM, MAX_UNITS, UNIT_FEAT_DIM,
    AlphaLiteNet, _combat_click_unit_legal,
)
from rl.roles import ROLE_VOCAB
from rl.tests.test_ra_completo_p0 import _B, _Obs, _U, _aidx


def test_harvesters_move_not_combat_push():
    assert "harvesters_move" not in COMBAT_PUSH_TYPES
    assert "harvest" not in COMBAT_PUSH_TYPES
    assert "army_attack_move" in COMBAT_PUSH_TYPES
    assert "attack_move" in COMBAT_PUSH_TYPES


def test_type_mask_off_when_safe_and_fresh():
    harv = _U(actor_id=11, type="harv", can_attack=False, cell_x=6, cell_y=6)
    obs = _Obs(units=[harv, _U(actor_id=20, type="e1")])
    aidx = _aidx(obs)
    assert not bool(aidx.type_mask[TYPE_TO_IDX["harvesters_move"]])
    assert not harvesters_need_move(obs)


def test_type_mask_on_when_harv_threatened():
    harv = _U(actor_id=11, type="harv", can_attack=False, cell_x=6, cell_y=6)
    ene = _U(actor_id=99, type="e1", cell_x=8, cell_y=6)
    obs = _Obs(units=[harv, _U(actor_id=20, type="e1")], enemies=[ene])
    aidx = _aidx(obs)
    assert harvesters_need_move(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["harvesters_move"]])


def test_type_mask_on_when_ore_stale():
    harv = _U(actor_id=11, type="harv", can_attack=False, cell_x=20, cell_y=20)
    obs = _Obs(units=[harv, _U(actor_id=20, type="e1")])
    # HWC spatial: rich ore next to proc (5,5), nothing under the harv.
    arr = np.zeros((32, 32, 9), dtype=np.float32)
    arr[:, :, 3] = 1.0
    arr[:, :, 4] = 1.0
    arr[6, 6, 2] = 8.0
    obs._spatial_chw = np.transpose(arr, (2, 0, 1)).copy()
    aidx = _aidx(obs)
    assert harvesters_need_move(obs)
    assert bool(aidx.type_mask[TYPE_TO_IDX["harvesters_move"]])


def test_remap_snaps_war_cell_to_proc():
    harv = _U(actor_id=11, type="harv", can_attack=False, cell_x=6, cell_y=6)
    obs = _Obs(units=[harv])
    aidx = _aidx(obs)
    x, y = remap_harvester_cell(obs, aidx, 28, 28)
    assert (x, y) == (5, 5)


def test_remap_keeps_cell_near_proc():
    harv = _U(actor_id=11, type="harv", can_attack=False, cell_x=6, cell_y=6)
    obs = _Obs(units=[harv])
    aidx = _aidx(obs)
    x, y = remap_harvester_cell(obs, aidx, 6, 5)
    assert (x, y) == (6, 5)


def test_harvest_cell_near_proc_kept():
    h1 = _U(actor_id=11, type="harv", can_attack=False, is_idle=True, cell_x=6, cell_y=6)
    obs = _Obs(units=[h1])
    aidx = _aidx(obs)
    cx, cy = 7, 6
    cell = cy * aidx.w + cx
    action, _eff = index_to_command_effective(
        obs, TYPE_TO_IDX["harvest"], aidx.unit_ids.index(11), cell, 0, aidx)
    cmd = action.commands[0]
    assert cmd.action == ActionType.HARVEST
    assert (cmd.target_x, cmd.target_y) == (7, 6)


def test_patrol_on_mcv_retargets_or_noop():
    mcv = _U(actor_id=3, type="mcv", can_attack=False, cell_x=4, cell_y=4)
    obs = _Obs(units=[mcv])
    aidx = _aidx(obs)
    cell = 5 * aidx.w + 5
    action, eff = index_to_command_effective(
        obs, TYPE_TO_IDX["patrol"], aidx.unit_ids.index(3), cell, 0, aidx)
    assert ACTION_TYPES[eff[0]] == "no_op"
    assert action.commands[0].action == ActionType.NO_OP


def test_unit_head_masks_harv_on_attack_move():
    net = AlphaLiteNet()
    B, U = 1, MAX_UNITS
    hidden = torch.zeros(B, HIDDEN_DIM)
    feats = torch.zeros(B, U, UNIT_FEAT_DIM)
    own = torch.zeros(B, U, dtype=torch.bool)
    own[0, 0] = True
    own[0, 1] = True
    feats[0, 1, 1] = 1.0  # rifle can_attack
    role = torch.zeros(B, U, dtype=torch.long)
    role[0, 0] = ROLE_VOCAB["harvester"]
    role[0, 1] = ROLE_VOCAB["infantry_basic"]
    t = torch.tensor([TYPE_TO_IDX["attack_move"]])
    legal = _combat_click_unit_legal(t, own, feats, role)
    assert not bool(legal[0, 0])
    assert bool(legal[0, 1])
    logits = net._scores_unit(hidden, t, feats, own, role)
    assert float(logits[0, 0]) < -1e3
    assert float(logits[0, 1]) > -1e3
    t_move = torch.tensor([TYPE_TO_IDX["move"]])
    legal_move = _combat_click_unit_legal(t_move, own, feats, role)
    assert bool(legal_move[0, 0]) and bool(legal_move[0, 1])


def test_harvesters_move_group_still_all_harvs():
    harvs = [
        _U(actor_id=11, type="harv", can_attack=False, cell_x=6, cell_y=6),
        _U(actor_id=12, type="harv", can_attack=False, cell_x=7, cell_y=6),
    ]
    obs = _Obs(units=harvs + [_U(actor_id=20, type="e1")])
    aidx = _aidx(obs)
    cell = 6 * aidx.w + 6
    action = index_to_command(
        obs, TYPE_TO_IDX["harvesters_move"], 0, cell, 0, aidx)
    assert sorted(c.actor_id for c in action.commands) == [11, 12]
    for c in action.commands:
        assert c.action == ActionType.MOVE
        assert (c.target_x, c.target_y) == (6, 6)


if __name__ == "__main__":
    test_harvesters_move_not_combat_push()
    test_type_mask_off_when_safe_and_fresh()
    test_type_mask_on_when_harv_threatened()
    test_type_mask_on_when_ore_stale()
    test_remap_snaps_war_cell_to_proc()
    test_remap_keeps_cell_near_proc()
    test_harvest_cell_near_proc_kept()
    test_patrol_on_mcv_retargets_or_noop()
    test_unit_head_masks_harv_on_attack_move()
    test_harvesters_move_group_still_all_harvs()
    print("OK harvester_safety tests")
