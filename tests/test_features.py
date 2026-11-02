import json

import numpy as np
import pytest

from ra_env import StepResult, pb
from ra_rl.features import (
    ALL_IN_OFFSET,
    CAND_DIM,
    FEATURE_VERSION,
    GLOBAL_DIM,
    HOLD,
    KEEP,
    MAX_TARGETS,
    N_ARMY_ACTIONS,
    N_ITEM_ACTIONS,
    PRIV_DIM,
    WAIT,
    Features,
    privileged_features,
    state_potential,
)
from ra_rl.policies.net import RandomPolicy
from ra_rl.policies.scripted import Params, ScriptedPolicy
from ra_rl.record import GameBuffer
from ra_rl.vocab import BUILDINGS, ITEM_INDEX, TARGET_KINDS


def game_obs(tick=100, faction="england") -> pb.GameObservation:
    o = pb.GameObservation(
        tick=tick,
        player_faction=faction,
        buildings=[pb.RlBuildingInfo(type="fact", cell_x=10, cell_y=10), pb.RlBuildingInfo(type="powr")],
        units=[
            pb.RlUnitInfo(actor_id=1, type="2tnk", cell_x=12, cell_y=10),
            pb.RlUnitInfo(actor_id=2, type="2tnk", cell_x=60, cell_y=60),
            pb.RlUnitInfo(actor_id=3, type="jeep", cell_x=40, cell_y=40),
        ],
        harass_unit_ids=[3],
        visible_enemies=[pb.RlUnitInfo(actor_id=7, type="3tnk", cell_x=62, cell_y=60)],
        visible_enemy_buildings=[pb.RlBuildingInfo(actor_id=9, type="barr", cell_x=70, cell_y=70)],
        production=[pb.RlProductionInfo(queue_type="Vehicle", item="2tnk")],
    )
    o.economy.cash, o.economy.power_provided, o.economy.power_drained = 1500, 100, 20
    return o


def army_req(*kinds, rid=3, queue="MAIN") -> pb.DecisionRequest:
    return pb.DecisionRequest(
        id=rid,
        kind=pb.DECIDE_ARMY,
        queue=queue,
        targets=[
            pb.TargetCandidate(
                kind=k,
                cell_x=60 + i,
                cell_y=60,
                type="barr" if k == "enemy_building" else ("harv" if k == "enemy_harvester" else ""),
            )
            for i, k in enumerate(kinds)
        ],
    )


def test_global_vector_is_deterministic_and_ignores_the_spectator_summary():
    assert FEATURE_VERSION == 4
    a, b = Features(), Features()
    a.observe(game_obs())
    o = game_obs()
    o.global_summary.enemy.cash = 99999
    o.global_summary.enemy.unit_value = 99999
    b.observe(o)
    assert a.g.shape == (GLOBAL_DIM,) and a.g.dtype == np.float32
    assert np.array_equal(a.g, b.g)
    assert np.isfinite(a.g).all()
    assert a.seen_enemy_soviet and not a.seen_enemy_allies
    assert a.harass_value > 0 and a.main_value > 0


def test_privileged_features_and_zero_sum_state_potential():
    o = game_obs()
    assert state_potential(o) == 0.0

    o.global_summary.own.unit_value = 2000
    o.global_summary.enemy.unit_value = 2000
    assert state_potential(o) == pytest.approx(0.0, abs=1e-6)

    o.global_summary.own.building_value = 5000
    o.global_summary.enemy.building_value = 1000
    phi_p1 = state_potential(o)
    assert 0.2 < phi_p1 <= 1.0

    # Mirror peer observation: exact zero-sum symmetry Phi(p0) == -Phi(p1)
    peer = game_obs(faction="russia")
    peer.global_summary.own.CopyFrom(o.global_summary.enemy)
    peer.global_summary.enemy.CopyFrom(o.global_summary.own)
    assert state_potential(peer) == pytest.approx(-phi_p1, abs=1e-6)

    p_sp = privileged_features(o, "selfplay")
    p_brutal = privileged_features(o, "brutal")
    assert p_sp.shape == (PRIV_DIM,) and np.isfinite(p_sp).all()
    assert not np.array_equal(p_sp, p_brutal)


def test_enemy_units_are_forgotten_buildings_remembered_longer_but_defenses_persist():
    f = Features()
    o = game_obs(tick=100)
    o.visible_enemy_buildings.append(pb.RlBuildingInfo(actor_id=42, type="tsla", cell_x=72, cell_y=70))
    f.observe(o)
    later = pb.GameObservation(tick=5000)
    f.observe(later)
    assert not f.enemy_units and not f.enemy_buildings
    assert 42 in f.seen_enemy_def_ids


def test_item_mask_label_and_decision_round_trip():
    f = Features()
    req = pb.DecisionRequest(id=1, kind=pb.DECIDE_TRAIN, queue="Vehicle", options=["harv", "2tnk", "zzz"])
    mask = f.item_mask(req)
    assert mask.shape == (N_ITEM_ACTIONS,) and mask.sum() == 3 and mask[WAIT]
    y = f.label(req, pb.Decision(request_id=1, choice="2tnk"))
    assert y == ITEM_INDEX["2tnk"] and f.decision(req, y).choice == "2tnk"
    assert f.label(req, pb.Decision(request_id=1, choice="")) == WAIT
    assert f.decision(req, WAIT).choice == ""
    assert f.label(req, pb.Decision(request_id=1, choice="zzz")) is None


def test_item_mask_sanity_caps_and_aircraft_rearm_rule():
    f = Features()
    o = pb.GameObservation(
        tick=2000,
        buildings=(
            [pb.RlBuildingInfo(type="proc")]
            + [pb.RlBuildingInfo(type="tent")] * 2
            + [pb.RlBuildingInfo(type="barr")] * 2
            + [pb.RlBuildingInfo(type="hpad")] * 1
            + [pb.RlBuildingInfo(type="afld.ukraine")] * 2
        ),
        units=[pb.RlUnitInfo(type="harv")] * 4 + [pb.RlUnitInfo(type="heli")] + [pb.RlUnitInfo(type="yak")] + [pb.RlUnitInfo(type="truk")] + [pb.RlUnitInfo(type="tran")],
    )
    f.observe(o)

    veh_req = pb.DecisionRequest(id=1, kind=pb.DECIDE_TRAIN, queue="Vehicle", options=["harv", "2tnk", "truk"])
    veh_mask = f.item_mask(veh_req)
    assert not veh_mask[ITEM_INDEX["harv"]]
    assert not veh_mask[ITEM_INDEX["truk"]]  # non-combat convoy capped at 1
    assert veh_mask[ITEM_INDEX["2tnk"]] and veh_mask[WAIT]

    bld_req = pb.DecisionRequest(id=2, kind=pb.DECIDE_BUILD, queue="Building", options=["tent", "barr", "weap", "sbag"])
    bld_mask = f.item_mask(bld_req)
    assert not bld_mask[ITEM_INDEX["tent"]] and not bld_mask[ITEM_INDEX["barr"]]
    assert not bld_mask[ITEM_INDEX["sbag"]]  # wall spam forbidden
    assert bld_mask[ITEM_INDEX["weap"]] and bld_mask[WAIT]

    # 1 hpad with 1 heli -> heli/mh60 capped; 2 afld with 1 yak -> yak/mig allowed
    air_req = pb.DecisionRequest(id=3, kind=pb.DECIDE_TRAIN, queue="Aircraft", options=["heli", "mh60", "yak", "mig", "tran"])
    air_mask = f.item_mask(air_req)
    assert not air_mask[ITEM_INDEX["heli"]] and not air_mask[ITEM_INDEX["mh60"]]
    assert not air_mask[ITEM_INDEX["tran"]]
    assert air_mask[ITEM_INDEX["yak"]] and air_mask[ITEM_INDEX["mig"]]


def test_consec_build_waits_breaks_fixed_point_state():
    f = Features()
    o = game_obs(tick=500)
    f.observe(o)
    g0 = f.g.copy()
    bld_req = pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["weap"])
    f.record(bld_req, pb.Decision(request_id=1, choice=""))
    f.observe(o)
    g1 = f.g.copy()
    assert f.consec_build_waits == 1
    assert not np.array_equal(g0, g1)

    f.record(bld_req, pb.Decision(request_id=1, choice="weap"))
    assert f.consec_build_waits == 0 and f.last_build_tick == 500


def test_multisquad_all_in_and_support_actions():
    f = Features()
    f.observe(game_obs())

    # MAIN supports normal attack (0..1), all_in (16..17), HOLD (32), KEEP (33)
    req_main = army_req("enemy_spawn", "enemy_building", queue="MAIN")
    cand, mask = f.candidates(req_main)
    assert cand.shape == (MAX_TARGETS, CAND_DIM) and mask.shape == (N_ARMY_ACTIONS,)
    assert mask[:2].all() and not mask[2:ALL_IN_OFFSET].any()
    assert mask[ALL_IN_OFFSET:ALL_IN_OFFSET + 2].all() and not mask[ALL_IN_OFFSET + 2:HOLD].any()
    assert mask[HOLD] and mask[KEEP]

    # Round-trip all_in on MAIN
    d_allin = f.decision(req_main, ALL_IN_OFFSET + 1)
    assert d_allin.choice == "all_in" and d_allin.target_index == 1
    assert f.label(req_main, d_allin) == ALL_IN_OFFSET + 1

    f.record(req_main, d_allin)
    assert f.army.attacking and f.army.all_in and f.army.target == ("enemy_building", 61, 60)
    f.observe(game_obs(tick=400))
    cand, _ = f.candidates(req_main)
    main_target_idx = len(TARGET_KINDS) + len(BUILDINGS) + 11
    assert cand[1, main_target_idx] == 1 and cand[0, main_target_idx] == 0

    # HARASS disables all_in (16..31) while keeping 0..1, HOLD, KEEP
    req_har = army_req("unexplored", "enemy_harvester", queue="HARASS")
    _, mask_har = f.candidates(req_har)
    assert mask_har[:2].all() and not mask_har[ALL_IN_OFFSET:HOLD].any()
    assert mask_har[HOLD] and mask_har[KEEP]

    # DECIDE_SUPPORT enables 0..k-1 and HOLD, disables all_in and KEEP
    req_sup = pb.DecisionRequest(
        id=9,
        kind=pb.DECIDE_SUPPORT,
        queue="SovietSpyPlane",
        targets=[pb.TargetCandidate(kind="unexplored", cell_x=50, cell_y=50)],
    )
    _, mask_sup = f.candidates(req_sup)
    assert mask_sup[0] and mask_sup[HOLD]
    assert not mask_sup[ALL_IN_OFFSET:HOLD].any() and not mask_sup[KEEP]
    assert f.decision(req_sup, 0).target_index == 0
    assert f.decision(req_sup, HOLD).target_index == -1


def test_random_policy_answers_validly():
    p = RandomPolicy()
    p.reset(3)
    build = pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["powr", "tent"])
    sup = pb.DecisionRequest(
        id=4, kind=pb.DECIDE_SUPPORT, queue="SovietSpyPlane",
        targets=[pb.TargetCandidate(kind="unexplored", cell_x=20, cell_y=20)],
    )
    for _ in range(20):
        step = StepResult(obs=game_obs(), tick=1, done=False, truncated=False, result="",
                          pending=[build, army_req("enemy_spawn"), sup])
        for d in p.decide(step):
            if d.request_id == 1:
                assert d.choice in ("", "powr", "tent")
            else:
                assert d.target_index in (-1, 0)


def test_scripted_sync_army_follows_the_executed_order():
    p = ScriptedPolicy(Params(attack_value=100))
    req = army_req("enemy_spawn", "unexplored")
    o = pb.GameObservation(units=[pb.RlUnitInfo(type="2tnk")])
    before = p.army_state()
    [d] = p.decide(StepResult(obs=o, tick=1, done=False, truncated=False, result="", pending=[req]))
    assert d.target_index == 0 and p.attacking
    p.sync_army(before, req, None, o)
    assert not p.attacking and p.target is None
    p.sync_army(p.army_state(), req, 1, o)
    assert p.attacking and p.target == ("unexplored", 61, 60)


def test_game_buffer_round_trip(tmp_path):
    f = Features()
    f.observe(game_obs())
    buf = GameBuffer()
    build = pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["powr"])
    buf.add(f, build, ITEM_INDEX["powr"])
    buf.add(f, army_req("enemy_spawn", queue="HARASS"), KEEP)
    path = tmp_path / "g000001.npz"
    buf.save(path, {"seed": 1, "outcome": "win"})
    z = np.load(path)
    assert z["prod_g"].shape == (1, GLOBAL_DIM) and z["prod_y"][0] == ITEM_INDEX["powr"]
    assert z["army_cand"].shape == (1, MAX_TARGETS, CAND_DIM) and z["army_y"][0] == KEEP
    assert z["army_group"][0] == 1  # HARASS channel index
    assert json.loads(str(z["meta"]))["outcome"] == "win"


def test_model_respects_masks_and_bc_loads(tmp_path):
    torch = pytest.importorskip("torch")
    from ra_rl.bc import load
    from ra_rl.model import CommanderNet

    f = Features()
    f.observe(game_obs())
    for seed in (1, 20):
        buf = GameBuffer()
        buf.add(f, pb.DecisionRequest(id=1, kind=pb.DECIDE_BUILD, queue="Building", options=["powr"]),
                ITEM_INDEX["powr"])
        buf.add(f, army_req("enemy_spawn"), 0)
        buf.save(tmp_path / f"g{seed:06d}.npz", {"seed": seed, "outcome": "win"})
    train, val, stats = load([str(tmp_path)])
    assert stats["games"] == 2 and len(train["prod_y"]) == 1 and len(val["army_y"]) == 1

    net = CommanderNet(hidden=32, cand_hidden=16)
    g = torch.from_numpy(train["prod_g"]).float()
    logits = net.item_logits(g, torch.from_numpy(train["prod_queue"]).long(), torch.from_numpy(train["prod_mask"]))
    assert logits.shape == (1, N_ITEM_ACTIONS) and int(logits.argmax()) in (ITEM_INDEX["powr"], WAIT)
    a = net.army_logits(
        g,
        torch.from_numpy(train["army_cand"]).float(),
        torch.from_numpy(train["army_mask"]),
        torch.from_numpy(train["army_group"]).long(),
    )
    assert a.shape == (1, N_ARMY_ACTIONS) and int(a.argmax()) in (0, ALL_IN_OFFSET, HOLD, KEEP)

    net.save(str(tmp_path / "m.pt"))
    from ra_rl.policies import make_policy
    for spec in (f"net:{tmp_path / 'm.pt'}", f"net-t0.25:{tmp_path / 'm.pt'}"):
        p = make_policy(spec)
        p.reset(1)
        step = StepResult(obs=game_obs(), tick=1, done=False, truncated=False, result="", pending=[army_req("stale")])
        assert all(d.target_index in (-1, 0) for d in p.decide(step))
