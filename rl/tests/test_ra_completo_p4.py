# -*- coding: utf-8 -*-
"""P4 RA-completo: obs scalars + role visibility for naval/air; dash combat split."""
import torch

from rl.network import AlphaLiteNet, adapt_scalar_state_dict
from rl.obs_encoding import (
    SCALAR_DIM,
    UNIT_FEAT_DIM,
    MAX_TOKENS,
    count_domain_units,
    is_air_type,
    is_naval_type,
    scalar_features,
    unit_tokens,
)
from rl.roles import role_id_of, role_of, ROLE_VOCAB


class _U:
    def __init__(self, typ="e1", aid=1, x=8, y=8):
        self.type = typ
        self.actor_id = aid
        self.cell_x = x
        self.cell_y = y
        self.hp_percent = 1.0
        self.can_attack = True
        self.is_idle = True
        self.speed = 50
        self.attack_range = 1000
        self.experience_level = 0
        self.stance = 0
        self.facing = 0


class _Obs:
    def __init__(self, units=None, enemies=None):
        self.units = units or []
        self.buildings = []
        self.visible_enemies = enemies or []
        self.visible_enemy_buildings = []
        self.production = []
        self.available_production = []
        self.economy = type("E", (), {
            "cash": 1000, "ore": 0, "resource_capacity": 2000,
            "power_provided": 100, "power_drained": 50, "harvester_count": 0,
        })()
        self.military = type("M", (), {
            "units_killed": 0, "units_lost": 0, "buildings_killed": 0,
            "buildings_lost": 0, "army_value": 0, "active_unit_count": 0,
            "kills_cost": 0, "deaths_cost": 0, "assets_value": 0,
        })()
        self.map_info = type("MI", (), {
            "height": 32, "width": 32, "map_name": "t",
        })()
        self.spatial_map = ""
        self.spatial_channels = 9
        self.tick = 1


def test_domain_type_helpers():
    assert is_naval_type("dd") and is_naval_type("ss") and is_naval_type("ca")
    assert is_air_type("heli") and is_air_type("mig") and is_air_type("tran")
    assert not is_naval_type("1tnk") and not is_air_type("1tnk")
    assert not is_naval_type("e1") and not is_air_type("e1")
    n, a = count_domain_units([_U("dd"), _U("heli"), _U("1tnk"), _U("e1")])
    assert (n, a) == (1, 1)


def test_scalar_dim_append_naval_air():
    assert SCALAR_DIM == 34
    obs = _Obs(
        units=[_U("dd", 1), _U("ca", 2), _U("heli", 3), _U("1tnk", 4)],
        enemies=[_U("ss", 10), _U("mig", 11), _U("yak", 12)],
    )
    sc = scalar_features(obs)
    assert sc.shape == (34,)
    # indices 29..32 = own_naval, own_air, ene_naval, ene_air (/10)
    assert abs(float(sc[29]) - 0.2) < 1e-5  # 2/10
    assert abs(float(sc[30]) - 0.1) < 1e-5  # 1/10
    assert abs(float(sc[31]) - 0.1) < 1e-5  # 1/10
    assert abs(float(sc[32]) - 0.2) < 1e-5  # 2/10
    # land-only still zeros the new cols
    sc0 = scalar_features(_Obs(units=[_U("e1"), _U("1tnk")]))
    assert float(sc0[29]) == 0.0 and float(sc0[30]) == 0.0
    assert float(sc0[31]) == 0.0 and float(sc0[32]) == 0.0
    # index 33 = signed power balance; default obs is 100-50 / 400 = 0.125
    assert abs(float(sc0[33]) - 0.125) < 1e-5


def test_role_ids_distinguish_naval_air_from_vehicle():
    """role_emb path: ship/heli/air get distinct ROLE_VOCAB ids vs land tank."""
    for t in ("dd", "ss", "ca", "pt", "lst", "msub"):
        assert "ship" in role_of(t) or role_of(t).startswith("ship")
    for t in ("heli", "mig", "yak", "hind", "tran", "badr"):
        r = role_of(t)
        assert r in ("heli", "air_fighter", "air_bomber", "transporter")
    tank_id = role_id_of("1tnk")
    assert role_id_of("dd") != tank_id
    assert role_id_of("heli") != tank_id
    assert role_id_of("dd") != role_id_of("heli")
    assert "ship_combat" in ROLE_VOCAB and "heli" in ROLE_VOCAB

    obs = _Obs(units=[_U("1tnk", 1), _U("dd", 2), _U("heli", 3)],
               enemies=[_U("ss", 9), _U("mig", 10)])
    feats, role_ids, valid, own = unit_tokens(obs)
    assert feats.shape == (MAX_TOKENS, UNIT_FEAT_DIM)
    assert int(role_ids[0]) == role_id_of("1tnk")
    assert int(role_ids[1]) == role_id_of("dd")
    assert int(role_ids[2]) == role_id_of("heli")
    # enemies packed after MAX_UNITS
    from rl.obs_encoding import MAX_UNITS
    assert int(role_ids[MAX_UNITS]) == role_id_of("ss")
    assert int(role_ids[MAX_UNITS + 1]) == role_id_of("mig")


def test_adapt_scalar_soft_pad_29_to_34():
    """Old SCALAR_DIM=29 ckpt pads new naval/air + power_balance cols to 0."""
    net = AlphaLiteNet()
    raw = net.state_dict()
    key = "scalar_mlp.0.weight"
    w = raw[key].clone()
    assert w.shape[1] == 34
    old = {k: v.clone() for k, v in raw.items()}
    old[key] = w[:, :29].contiguous()
    adapted = adapt_scalar_state_dict(net, old)
    aw = adapted[key]
    assert aw.shape == w.shape
    assert torch.allclose(aw[:, :29], w[:, :29])
    assert torch.allclose(aw[:, 29:], torch.zeros_like(aw[:, 29:]))
    net2 = AlphaLiteNet()
    net2.load_state_dict(adapted, strict=False)
    assert net2.scalar_mlp[0].weight.shape[1] == 34


def test_adapt_scalar_soft_pad_33_to_34():
    """best_B / ck103 (in=33) pads power_balance col to 0."""
    net = AlphaLiteNet()
    raw = net.state_dict()
    key = "scalar_mlp.0.weight"
    w = raw[key].clone()
    old = {k: v.clone() for k, v in raw.items()}
    old[key] = w[:, :33].contiguous()
    adapted = adapt_scalar_state_dict(net, old)
    aw = adapted[key]
    assert aw.shape == w.shape
    assert torch.allclose(aw[:, :33], w[:, :33])
    assert torch.allclose(aw[:, 33:], torch.zeros_like(aw[:, 33:]))


def test_dashboard_attack_keys_documented():
    """dashboard.html must count naval/air separately (not under vehicle)."""
    from pathlib import Path
    html = Path(__file__).resolve().parents[2].joinpath("dashboard.html").read_text(
        encoding="utf-8")
    assert "naval_attack_move" in html
    assert "air_attack_move" in html
    assert "navalCombatFrac" in html and "airCombatFrac" in html
    assert "landCombatFrac" in html
    # must not be the old land-only ATK set without navy/air
    assert "NAVAL_ATK" in html and "AIR_ATK" in html
