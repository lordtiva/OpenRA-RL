# -*- coding: utf-8 -*-
"""Tests del shaper con incentivo minero + invariante de suma exacta."""
import sys

sys.path.insert(0, ".")

from types import SimpleNamespace as NS
from rl.reward_shaping import ShapedReward
from rl.action_adapter import _split_production


def obs(cash=5000, harv=0, kills=0, deaths=0, assets=2000,
        bldgs=("conyard",), prod=()):
    buildings = [NS(type=t, actor_id=100 + i, cell_x=10, cell_y=5)
                 for i, t in enumerate(bldgs)]
    return NS(
        tick=100,
        economy=NS(cash=cash, ore=0, harvester_count=harv,
                   power_provided=100, power_drained=60, resource_capacity=5000),
        military=NS(kills_cost=kills, deaths_cost=deaths,
                    assets_value=assets, units_killed=0, units_dead=0),
        buildings=buildings,
        production=list(prod),
    )


# 1) refineria nueva paga refineria + cosechador incluido
sh = ShapedReward()
sh.reset(obs(harv=0))
o2 = obs(harv=1, assets=4000, bldgs=("conyard", "proc"))
r = sh.step(o2, done=False)
comp = sh.last_components
assert comp["mining"] == 1.0 + 0.25, comp          # proc (1.0) + harv (0.25)
assert abs(sum(comp.values()) - r) < 1e-9           # INVARIANTE suma exacta
print(f"1) primera proc: mining={comp['mining']} | suma OK")

# 2) nacer con proc NO paga nada (reset marca como ya pagada)
sh2 = ShapedReward()
sh2.reset(obs(harv=1, bldgs=("conyard", "proc")))
r2 = sh2.step(obs(harv=1, assets=2100), done=False)
assert sh2.last_components["mining"] == 0.0
print("2) proc preexistente no paga: OK")

# 3) cosechadoras hasta el tope, despues nada; muerte de cosechadora no roba
sh3 = ShapedReward()
sh3.reset(obs(harv=1))
sh3.step(obs(harv=4, assets=3000), done=False)      # +3 cosechadoras pagadas
m3 = sh3.last_components["mining"]
assert m3 == 0.75, m3                               # 3 x 0.25
sh3.step(obs(harv=6, assets=3500), done=False)      # por encima del tope: $0
assert sh3.last_components["mining"] == 0.75
sh3.step(obs(harv=3, assets=3300), done=False)      # murieron 3: sin castigo
assert sh3.last_components["mining"] == 0.75
print("3) tope de cosechadoras y no-robo tras muerte: OK")

# 4) split estatico train/build -> AHORA roles (traductor universal por facción)
class B:
    def __init__(self, can):
        self.can_produce = can
        self.type = "x"

obs_split = NS(available_production=["e1", "dog", "proc", "powr", "barr"],
               buildings=[B(["e1", "proc"])])
tr, bu, rol_conc = _split_production(obs_split)
# edificios al cubo build (por rol): proc/powr/barr refinan|power|barracks
assert bu == ["barracks", "power", "refinery"], bu
# unidades al cubo train (por rol): dog->infantry_antiinf, e1->infantry_basic
assert tr == ["infantry_antiinf", "infantry_basic"], tr
assert rol_conc["refinery"] == "proc", rol_conc
assert rol_conc["infantry_basic"] == "e1", rol_conc
print("4) split por ROL (agnóstico a facción): train=", tr, "build=", bu)

print("\nTODOS LOS TESTS OK")
