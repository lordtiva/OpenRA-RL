# -*- coding: utf-8 -*-
"""Anti-rush mask: weap + 2nd proc wait for barracks and 4 combat."""
from types import SimpleNamespace as NS

from openra_env.models import ActionType
from rl.action_adapter import (
    ANTI_RUSH_COMBAT,
    ActionIndex,
    Vocab,
    anti_rush_unlocked,
    index_to_command_effective,
    n_proc_count,
    owns_barracks,
)
from rl.auto_support import (
    SUPPORT_FOG_SCOUT,
    SUPPORT_LATE_REMNANT,
    SUPPORT_WAR_NUDGE,
    support_commands,
)
from rl.network import TYPE_TO_IDX


def _u(actor_id=1, typ="e1", x=12, y=16, idle=True):
    return NS(actor_id=actor_id, type=typ, cell_x=x, cell_y=y,
              is_idle=idle, hp_percent=1.0, can_attack=True)


def _b(typ="fact", actor_id=10, x=12, y=16):
    return NS(type=typ, actor_id=actor_id, cell_x=x, cell_y=y,
              hp_percent=1.0, is_repairing=False, is_powered=True,
              rally_x=-1, rally_y=-1)


def _obs(*, cash=5000, harv=1, bldgs=("fact", "proc"), units=None, prod=(),
         avail=("e1", "harv", "proc", "powr", "tent", "weap", "barr")):
    if units is None:
        units = [_u(9, "harv", 14, 16)]
    return NS(
        tick=1000,
        map_info=NS(height=64, width=128, map_name="a_short"),
        economy=NS(cash=cash, ore=0, harvester_count=harv,
                   power_provided=100, power_drained=60, resource_capacity=5000),
        military=NS(kills_cost=0, deaths_cost=0, assets_value=2000,
                    units_killed=0, units_dead=0, army_value=0),
        buildings=[_b(t, 100 + i) for i, t in enumerate(bldgs)],
        units=list(units),
        production=list(prod),
        available_production=list(avail),
        visible_enemies=[],
        visible_enemy_buildings=[],
        spatial_map="",
        spatial_channels=9,
    )


def _slot(aidx, role):
    return aidx.items.index(role)


def test_constants():
    assert ANTI_RUSH_COMBAT == 4
    assert SUPPORT_WAR_NUDGE is False
    assert SUPPORT_FOG_SCOUT is False
    assert SUPPORT_LATE_REMNANT is False


def test_unlock_needs_barracks_and_four_combat():
    eco = _obs(bldgs=("fact", "proc"), units=[_u(9, "harv")])
    assert owns_barracks(eco) is False
    assert anti_rush_unlocked(eco) is False
    tent_only = _obs(bldgs=("fact", "proc", "tent"), units=[_u(9, "harv")])
    assert owns_barracks(tent_only) is True
    assert anti_rush_unlocked(tent_only) is False
    four = [_u(i, "e1") for i in range(1, 5)] + [_u(9, "harv")]
    ready = _obs(bldgs=("fact", "proc", "tent"), units=four)
    assert anti_rush_unlocked(ready) is True
    kenn = _obs(bldgs=("fact", "proc", "kenn"), units=four)
    assert owns_barracks(kenn) is False
    assert anti_rush_unlocked(kenn) is False


def test_weap_masked_until_unlock():
    obs = _obs(bldgs=("fact", "proc", "tent"), units=[_u(9, "harv")])
    aidx = ActionIndex(obs, Vocab())
    assert "warf" in aidx.build_items
    bslot = len(aidx.train_items) + aidx.build_items.index("warf")
    assert bool(aidx.build_slot_mask[bslot]) is False
    act, _ = index_to_command_effective(
        obs, TYPE_TO_IDX["build"], 0, 0, _slot(aidx, "warf"), aidx)
    assert act.commands[0].action.value == "no_op"


def test_second_proc_masked_until_unlock():
    obs = _obs(bldgs=("fact", "proc"), units=[_u(9, "harv")])
    assert n_proc_count(obs) == 1
    aidx = ActionIndex(obs, Vocab())
    rslot = len(aidx.train_items) + aidx.build_items.index("refinery")
    assert bool(aidx.build_slot_mask[rslot]) is False
    act, _ = index_to_command_effective(
        obs, TYPE_TO_IDX["build"], 0, 0, _slot(aidx, "refinery"), aidx)
    assert act.commands[0].action.value == "no_op"


def test_first_proc_and_barracks_stay_legal():
    obs = _obs(bldgs=("fact", "powr"), units=[_u(1, "mcv")], harv=0,
               avail=("proc", "powr", "tent", "weap"))
    aidx = ActionIndex(obs, Vocab())
    rslot = len(aidx.train_items) + aidx.build_items.index("refinery")
    assert bool(aidx.build_slot_mask[rslot]) is True
    tent = _obs(bldgs=("fact", "proc"), units=[_u(9, "harv")])
    aidx_t = ActionIndex(tent, Vocab())
    bslot = len(aidx_t.train_items) + aidx_t.build_items.index("barracks")
    assert bool(aidx_t.build_slot_mask[bslot]) is True


def test_harv_train_not_capped():
    obs = _obs(bldgs=("fact", "proc", "weap"),
               units=[_u(9, "harv"), _u(10, "harv"), _u(11, "harv")],
               harv=3)
    # still locked (no barracks / 4 combat) but harv TRAIN stays on
    aidx = ActionIndex(obs, Vocab())
    if "harvester" in aidx.train_items:
        slot = aidx.train_items.index("harvester")
        assert bool(aidx.train_slot_mask[slot]) is True


def test_unlocked_weap_and_second_proc_legal():
    four = [_u(i, "e1") for i in range(1, 5)] + [_u(9, "harv")]
    obs = _obs(bldgs=("fact", "proc", "tent"), units=four)
    aidx = ActionIndex(obs, Vocab())
    wslot = len(aidx.train_items) + aidx.build_items.index("warf")
    rslot = len(aidx.train_items) + aidx.build_items.index("refinery")
    assert bool(aidx.build_slot_mask[wslot]) is True
    assert bool(aidx.build_slot_mask[rslot]) is True
    act_w, _ = index_to_command_effective(
        obs, TYPE_TO_IDX["build"], 0, 0, _slot(aidx, "warf"), aidx)
    assert act_w.commands[0].action.value == "build"
    assert act_w.commands[0].item_type in ("weap", "warf")
    act_p, _ = index_to_command_effective(
        obs, TYPE_TO_IDX["build"], 0, 0, _slot(aidx, "refinery"), aidx)
    assert act_p.commands[0].action.value == "build"
    assert act_p.commands[0].item_type in ("proc", "refinery")


def test_queued_proc_counts_as_first():
    prod = [NS(queue_type="Building", item="proc", progress=0.4, paused=False)]
    obs = _obs(bldgs=("fact",), units=[_u(1, "mcv")], harv=0, prod=prod)
    assert n_proc_count(obs) == 1


def _kinds(cmds):
    return [(c.action.value, c.item_type) for c in cmds]


def test_support_no_strategy_commands():
    cmds = support_commands(_obs(
        cash=5000, bldgs=("fact", "proc"),
        avail=("e1", "harv", "proc", "powr", "tent", "weap"),
        units=[_u(9, "harv", idle=True)]))
    forbidden = {
        "build", "place_building", "train", "deploy",
        "attack_move", "army_attack_move", "set_rally_point",
        "set_stance", "sell",
    }
    assert not any(c.action.value in forbidden for c in cmds)
    assert any(c.action.value == "harvest" for c in cmds)
    assert all(
        c.action.value != "harvest" or (int(c.target_x or 0) == 0
                                        and int(c.target_y or 0) == 0)
        for c in cmds)


def test_support_repair_and_power_down():
    obs = _obs(cash=2000, bldgs=("fact", "proc", "dome"))
    obs.buildings[0].hp_percent = 0.2
    obs.economy.power_provided = 10
    obs.economy.power_drained = 80
    obs.buildings[-1].type = "dome"
    obs.buildings[-1].is_powered = True
    cmds = support_commands(obs)
    assert any(c.action.value == "repair" for c in cmds)
    assert any(c.action.value == "power_down" for c in cmds)
