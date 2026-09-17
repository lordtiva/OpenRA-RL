# -*- coding: utf-8 -*-
"""Fog leave must not pay w_raze_prod; real kills/value drops may."""
from types import SimpleNamespace as NS

from rl.reward_shaping import ShapedReward


class _Mil:
    def __init__(self, kills=0, deaths=0, assets=0, bk=0):
        self.kills_cost = kills
        self.deaths_cost = deaths
        self.assets_value = assets
        self.buildings_killed = bk
        self.units_killed = 0
        self.units_lost = 0
        self.army_value = 0
        self.active_unit_count = 0


def _obs(ene_prod_types=(), buildings_killed=0, cash=1000, kills=0, deaths=0):
    ene = [NS(type=t, cell_x=1, cell_y=1) for t in ene_prod_types]
    return NS(
        tick=5000,
        buildings=[NS(type="fact"), NS(type="proc"), NS(type="tent")],
        units=[],
        military=_Mil(kills=kills, deaths=deaths, bk=buildings_killed),
        economy=NS(cash=cash, ore=0, harvester_count=1,
                   power_provided=100, power_drained=50,
                   resource_capacity=2000),
        visible_enemy_buildings=ene,
        visible_enemies=[],
        result="",
    )


def _gs(ebv=8000, n_ene=5):
    return {
        "own": {"cash": 1000, "unit_value": 0, "building_value": 5000,
                "n_buildings": 3, "earned": 0},
        "enemy": {"cash": 1000, "unit_value": 0, "building_value": ebv,
                  "n_buildings": n_ene, "earned": 0},
    }


def test_fog_leave_no_raze_prod_bonus():
    sh = ShapedReward("eradicate_v5")
    o0 = _obs(ene_prod_types=("weap",), buildings_killed=0)
    sh.reset(o0)
    gs = _gs()
    sh.step(o0, done=False, gs=gs)
    raze_before = float(sh.last_components.get("raze", 0.0))
    o1 = _obs(ene_prod_types=(), buildings_killed=0)
    sh.step(o1, done=False, gs=_gs())
    assert float(sh.last_components.get("raze", 0.0)) <= raze_before + 1e-6, (
        sh.last_components)


def test_sticky_episode_raze_does_not_gate_fog_leave():
    """Prior-step turret raze must not unlock fog-leave prod phantom pay."""
    sh = ShapedReward("eradicate_v5")
    o0 = _obs(ene_prod_types=("weap", "weap"), buildings_killed=0)
    sh.reset(o0)
    sh.step(o0, done=False, gs=_gs(ebv=8000, n_ene=6))
    # Real non-prod value drop (turret): pays base raze, no prod drop yet
    o1 = _obs(ene_prod_types=("weap", "weap"), buildings_killed=0)
    sh.step(o1, done=False, gs=_gs(ebv=7500, n_ene=5))
    assert float(sh.last_components.get("raze", 0.0)) > 0.0
    raze_after_turret = float(sh.last_components["raze"])
    # Later fog leave: both weap vanish from vision, bv unchanged, no kill
    o2 = _obs(ene_prod_types=(), buildings_killed=0)
    sh.step(o2, done=False, gs=_gs(ebv=7500, n_ene=5))
    assert float(sh.last_components.get("raze", 0.0)) <= raze_after_turret + 1e-6, (
        sh.last_components)


def test_real_prod_kill_pays_bonus():
    sh = ShapedReward("eradicate_v5")
    o0 = _obs(ene_prod_types=("weap",), buildings_killed=0)
    sh.reset(o0)
    sh.step(o0, done=False, gs=_gs(ebv=8000))
    o1 = _obs(ene_prod_types=(), buildings_killed=1)
    sh.step(o1, done=False, gs=_gs(ebv=6000, n_ene=4))
    assert float(sh.last_components.get("raze", 0.0)) > 0.4, sh.last_components


def test_v5_exchange_off_keeps_death_magnitude():
    """Losing $100 vs $1000 must not collapse to the same ±w_exchange."""
    sh = ShapedReward("eradicate_v5")
    assert float(sh.w_exchange) == 0.0
    o0 = _obs()
    sh.reset(o0)
    gs = _gs()
    sh.step(o0, done=False, gs=gs)
    # Lose e1 ($100)
    sh.step(_obs(deaths=100), done=False, gs=gs)
    e1 = float(sh.last_components.get("combat", 0.0))
    sh.reset(o0)
    sh.step(o0, done=False, gs=gs)
    sh.step(_obs(deaths=1000), done=False, gs=gs)
    tank = float(sh.last_components.get("combat", 0.0))
    assert e1 < 0 and tank < 0
    assert abs(tank) > abs(e1) * 5, (e1, tank)
