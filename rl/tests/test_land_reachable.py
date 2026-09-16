# -*- coding: utf-8 -*-
"""Unit tests: hunt con conectividad terrestre (A) y anti-stuck (B).

El hunt ciego ordenaba niebla al otro lado del lago (a_short oeste:
20 rifles idle 54k ticks sin contacto). Sin spatial, todo se comporta
como antes.
"""
from __future__ import annotations

import base64 as _b64
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rl.auto_support import (
    fog_scout_destinations,
    hunt_near_cell,
    land_connected,
    nearest_reachable,
)
from rl.war_objective import war_objective

ok = True


def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    ok = ok and bool(cond)


def _u(actor_id=1, typ="e1", x=12, y=16, idle=True):
    return NS(actor_id=actor_id, type=typ, cell_x=x, cell_y=y,
              is_idle=idle, hp_percent=1.0, can_attack=True)


def _b(typ="fact", actor_id=10, x=12, y=16):
    return NS(type=typ, actor_id=actor_id, cell_x=x, cell_y=y,
              hp_percent=1.0, is_repairing=False, is_powered=True,
              rally_x=-1, rally_y=-1)


def _lake_obs(w=32, h=32):
    """Mapa 32x32 partido por un lago vertical (x=15,16): este|oeste."""
    ch = 9
    arr = np.zeros((h, w, ch), dtype=np.float32)
    arr[:, :, 3] = 1.0  # todo pasable...
    arr[:, 15:17, 3] = 0.0  # ...menos el lago
    arr[:, :, 4] = 0.0  # todo niebla
    spatial = _b64.b64encode(arr.tobytes()).decode("ascii")
    west_army = [_u(i, "e1", 5 + (i % 3), 16) for i in range(1, 9)]
    return NS(
        tick=1000,
        map_info=NS(height=h, width=w, map_name="test_lake.oramap"),
        economy=NS(cash=2000, ore=0, harvester_count=1,
                   power_provided=100, power_drained=60,
                   resource_capacity=5000),
        military=NS(kills_cost=0, deaths_cost=0, assets_value=2000,
                    units_killed=0, units_dead=0, army_value=0),
        buildings=[_b("fact", 100, 5, 16), _b("proc", 101, 6, 17),
                   _b("tent", 102, 5, 18)],
        units=west_army + [_u(99, "harv", 6, 16)],
        production=[],
        available_production=["e1"],
        visible_enemies=[],
        visible_enemy_buildings=[],
        spatial_map=spatial,
        spatial_channels=ch,
        global_summary=None,
    )


obs = _lake_obs()

check("mismo lado conectado", land_connected(obs, (5, 16), (10, 20)))
check("otro lado NO conectado", not land_connected(obs, (5, 16), (25, 20)))
check("agua NO conecta", not land_connected(obs, (5, 16), (15, 20)))
check("sin terreno → True (legacy)",
      land_connected(NS(), (0, 0), (100, 100)))

nr = nearest_reachable(obs, (25, 20), (5, 16))
check("nearest_reachable cae de este lado",
      nr is not None and nr[0] < 15)
check("nearest_reachable en destino válido = identidad",
      nearest_reachable(obs, (10, 20), (5, 16)) == (10, 20))

dests = fog_scout_destinations(obs, 4, from_xy=(5, 16))
check("fog dests no vacías", len(dests) > 0)
check("fog dests todas de este lado (x<15)",
      all(x < 15 for x, y in dests))

hunt = hunt_near_cell(obs, (25, 20), from_xy=(5, 16))
check("hunt al otro lado se reubica de este lado", hunt[0] < 15)
hunt_same = hunt_near_cell(obs, (10, 20), from_xy=(5, 16))
check("hunt válido sigue cerca del ancla y de este lado",
      abs(hunt_same[0] - 10) + abs(hunt_same[1] - 20) <= 24
      and hunt_same[0] < 15)

obj = war_objective(obs, from_xy=(5, 16))
check("war_objective fog cae de este lado",
      obj is not None and obj[0] < 15)

# Sin spatial: comportamiento previo (sin filtro, sin crash).
bare = _lake_obs()
bare.spatial_map = ""
dests_bare = fog_scout_destinations(bare, 2)
check("sin spatial: fog igual devuelve", len(dests_bare) == 2)
hunt_bare = hunt_near_cell(bare, (25, 20))
check("sin spatial: hunt no crashea y da celda",
      isinstance(hunt_bare, tuple) and len(hunt_bare) == 2)

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS"))
sys.exit(0 if ok else 1)
