from ra_env import StepResult, pb
from ra_rl.eval import wilson
from ra_rl.policies import BuiltinPolicy, make_policy
from ra_rl.policies.scripted import Params, ScriptedPolicy


def obs(buildings=(), units=(), power=(100, 0), tick=100, faction="") -> pb.GameObservation:
    o = pb.GameObservation(
        tick=tick,
        player_faction=faction,
        buildings=[pb.RlBuildingInfo(type=t, cell_x=10, cell_y=10) for t in buildings],
        units=[pb.RlUnitInfo(actor_id=i + 1, type=t, cell_x=12, cell_y=10) for i, t in enumerate(units)],
    )
    o.economy.power_provided, o.economy.power_drained = power
    return o


def step(o: pb.GameObservation, *requests: pb.DecisionRequest) -> StepResult:
    return StepResult(obs=o, tick=o.tick, done=False, truncated=False, result="", pending=list(requests))


def build_req(*options, queue="Building", rid=1):
    return pb.DecisionRequest(id=rid, kind=pb.DECIDE_BUILD, queue=queue, options=list(options))


def train_req(queue, *options, rid=2):
    return pb.DecisionRequest(id=rid, kind=pb.DECIDE_TRAIN, queue=queue, options=list(options))


def army_req(*targets, rid=3, queue="MAIN"):
    return pb.DecisionRequest(
        id=rid,
        kind=pb.DECIDE_ARMY,
        queue=queue,
        targets=[pb.TargetCandidate(kind=k, cell_x=i * 25, cell_y=i * 25) for i, k in enumerate(targets)],
    )


def test_make_policy():
    assert make_policy("builtin").brain == "builtin"
    assert make_policy("scripted").brain == "external"
    assert BuiltinPolicy().decide(step(obs(), build_req("powr"))) == []


def test_build_follows_order_for_allies_and_soviet():
    p = ScriptedPolicy()
    [d] = p.decide(step(obs(buildings=["fact"]), build_req("powr", "tent", "proc")))
    assert (d.request_id, d.choice) == (1, "powr")
    [d] = p.decide(step(obs(buildings=["fact", "powr"]), build_req("powr", "tent", "proc")))
    assert d.choice == "tent"

    # Soviet build order (`barr`, `afld.ukraine`, `tsla`)
    p_sov = ScriptedPolicy()
    [d_s] = p_sov.decide(step(obs(buildings=["fact", "powr"], faction="ukraine"), build_req("powr", "barr", "proc")))
    assert d_s.choice == "barr"
    [d_af] = p_sov.decide(step(
        obs(buildings=["fact", "powr", "powr", "barr", "proc", "proc", "proc", "weap", "weap", "dome", "apwr", "apwr"], power=(600, 200)),
        build_req("powr", "apwr", "afld.ukraine", "stek"),
    ))
    assert d_af.choice == "afld.ukraine"


def test_build_powers_up_before_a_consumer():
    p = ScriptedPolicy()
    [d] = p.decide(step(obs(buildings=["fact", "powr", "tent"], power=(100, 80)), build_req("powr", "proc", "tent")))
    assert d.choice == "powr"


def test_defense_waits_for_war_factory_unless_early_rush_with_refinery():
    p = ScriptedPolicy()
    [d] = p.decide(step(obs(buildings=["fact"]), build_req("pbox", queue="Defense")))
    assert d.choice == ""
    [d] = p.decide(step(obs(buildings=["fact", "weap"]), build_req("pbox", queue="Defense")))
    assert d.choice == "pbox"

    p2 = ScriptedPolicy()
    rush_obs = obs(buildings=["fact", "powr", "tent", "proc"], tick=2500)
    rush_obs.visible_enemies.extend([pb.RlUnitInfo(type="e1", cell_x=14, cell_y=10)] * 8)
    [d] = p2.decide(step(rush_obs, build_req("pbox", queue="Defense")))
    assert d.choice == "pbox"


def test_train_harvesters_tanks_artillery_and_aircraft_rearm_rule():
    p = ScriptedPolicy(Params(harv_per_proc=2, arty_every=4, siege_arty_every=2, siege_def_threshold=3))
    opts = ("harv", "1tnk", "2tnk", "arty")
    [d] = p.decide(step(obs(buildings=["proc"], units=["harv"]), train_req("Vehicle", *opts)))
    assert d.choice == "harv"
    [d] = p.decide(step(obs(buildings=["proc"], units=["harv"] * 2), train_req("Vehicle", *opts)))
    assert d.choice == "2tnk"
    [d] = p.decide(step(obs(buildings=["proc"], units=["harv"] * 2 + ["2tnk"] * 4), train_req("Vehicle", *opts)))
    assert d.choice == "arty"

    # Strict Rearm Rule: 0 pads -> "", 1 hpad & 0 helis -> "heli", 1 hpad & 1 heli -> ""
    [d] = p.decide(step(obs(), train_req("Aircraft", "heli")))
    assert d.choice == ""
    [d] = p.decide(step(obs(buildings=["hpad"]), train_req("Aircraft", "heli", "mh60")))
    assert d.choice == "heli"
    [d] = p.decide(step(obs(buildings=["hpad"], units=["heli"]), train_req("Aircraft", "heli", "mh60")))
    assert d.choice == ""

    # Spotting >=3 static defenses activates siege artillery ratio (1 arty/v2rl per 2 tanks)
    turtle_obs = obs(buildings=["proc"], units=["harv"] * 2 + ["3tnk"] * 2)
    for aid in (10, 11, 12):
        turtle_obs.visible_enemy_buildings.append(pb.RlBuildingInfo(actor_id=aid, type="tsla"))
    [d] = p.decide(step(turtle_obs, train_req("Vehicle", "harv", "3tnk", "v2rl")))
    assert d.choice == "v2rl"


def test_multisquad_all_in_harass_air_and_support():
    p = ScriptedPolicy(Params(attack_value=1000, all_in_value=3000))
    # Overwhelming MAIN force on enemy_building triggers choice="all_in"
    big_obs = obs(buildings=["fact"], units=["4tnk"] * 3)
    [d_main] = p.decide(step(big_obs, army_req("enemy_spawn", "enemy_building", queue="MAIN")))
    assert d_main.target_index == 1 and d_main.choice == "all_in"

    # HARASS prioritizes unexplored when explored_percent < 45%, then enemy_harvester
    low_exp = obs(units=["jeep"])
    low_exp.explored_percent = 20.0
    [d_har1] = p.decide(step(low_exp, army_req("enemy_building", "enemy_harvester", "unexplored", queue="HARASS")))
    assert d_har1.target_index == 2 and d_har1.choice == "attack"

    p2 = ScriptedPolicy()
    high_exp = obs(units=["ftrk"])
    high_exp.explored_percent = 65.0
    [d_har2] = p2.decide(step(high_exp, army_req("enemy_building", "enemy_harvester", "unexplored", queue="HARASS")))
    assert d_har2.target_index == 1 and d_har2.choice == "attack"

    # AIR avoids AA cluster around target 0 and strikes undefended harvester at target 1
    p_air = ScriptedPolicy()
    aa_obs = obs(units=["heli"])
    aa_obs.visible_enemy_buildings.extend([
        pb.RlBuildingInfo(actor_id=50, type="sam", cell_x=1, cell_y=1),
        pb.RlBuildingInfo(actor_id=51, type="sam", cell_x=2, cell_y=2),
    ])
    [d_air] = p_air.decide(step(aa_obs, army_req("enemy_harvester", "enemy_harvester", queue="AIR")))
    assert d_air.target_index == 1

    # DECIDE_SUPPORT: SpyPlane scouts unexplored; Iron Curtain waits when MAIN is not attacking
    sup_spy = pb.DecisionRequest(
        id=8, kind=pb.DECIDE_SUPPORT, queue="SovietSpyPlane",
        targets=[pb.TargetCandidate(kind="enemy_building", cell_x=10, cell_y=10),
                 pb.TargetCandidate(kind="unexplored", cell_x=80, cell_y=80)],
    )
    [d_spy] = p2.decide(step(high_exp, sup_spy))
    assert d_spy.target_index == 1

    sup_ic = pb.DecisionRequest(
        id=9, kind=pb.DECIDE_SUPPORT, queue="GrantExternalConditionPowerInfoOrder",
        targets=[pb.TargetCandidate(kind="enemy_building", cell_x=10, cell_y=10)],
    )
    [d_ic] = p2.decide(step(high_exp, sup_ic))
    assert d_ic.target_index == -1 and d_ic.choice == "hold"


def test_army_attacks_keeps_reinforces_and_recalls():
    p = ScriptedPolicy(Params(attack_value=1000, reinforce_value=800, recall_fraction=0.5))
    req = army_req("enemy_spawn", "enemy_building")
    away = [pb.RlUnitInfo(type="2tnk", cell_x=90, cell_y=10)] * 2
    yard = [pb.RlBuildingInfo(type="fact", cell_x=10, cell_y=40)]

    def at(units):
        return pb.GameObservation(buildings=yard, units=units)

    assert p.decide(step(at(away[:1]), req)) == []
    [d] = p.decide(step(at(away), req))
    assert d.target_index == 1
    assert p.decide(step(at(away), req)) == []
    home = [pb.RlUnitInfo(type="2tnk", cell_x=12, cell_y=40)] * 1
    [d] = p.decide(step(at(away + home), req))
    assert d.target_index == 1
    [d] = p.decide(step(at(away[:1]), req))
    assert d.target_index == -1


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(60, 100)
    assert 0.50 < lo < 0.51 and 0.69 < hi < 0.70


def test_army_moves_on_when_target_disappears_and_gives_up_on_stuck_target():
    p = ScriptedPolicy(Params(attack_value=100, reinforce_value=10**9, max_aim_decisions=2))
    req = army_req("unexplored", "unexplored")
    [d] = p.decide(step(obs(units=["2tnk"]), req))
    assert d.target_index == 0
    assert p.decide(step(obs(units=["2tnk"]), req)) == []
    [d] = p.decide(step(obs(units=["2tnk"]), req))
    assert d.target_index == 1
    assert p.decide(step(obs(units=["2tnk"]), req)) == []
    [d] = p.decide(step(obs(units=["2tnk"]), req))
    assert d.target_index == 0
