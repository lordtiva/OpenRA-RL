# -*- coding: utf-8 -*-
"""Smoke tests for live viewer lobby flags (faction + spawn map patch)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from rl.live_lobby import (
    classify_spawn_indices,
    list_mpspawn_cells,
    normalize_enemy_faction,
    normalize_player_faction,
    normalize_spawn,
    patch_oramap_bytes,
    resolve_episode_spawn,
)


ROOT = Path(__file__).resolve().parents[1]
A_SHORT = ROOT / "scenarios" / "fase2_a_short.oramap"


def test_normalize_lobby_flags():
    assert normalize_spawn("NE") == "ne"
    assert normalize_spawn("1") == "sw"
    assert normalize_spawn("random") == "random"
    assert normalize_player_faction("france") == "france"
    assert normalize_player_faction("Random") == "RandomAllies"
    assert normalize_enemy_faction("") == "Random"
    assert normalize_enemy_faction("russia") == "russia"
    with pytest.raises(ValueError):
        normalize_player_faction("russia")
    with pytest.raises(ValueError):
        normalize_spawn("east")


def test_resolve_episode_spawn_random_rotates():
    import random
    rng = random.Random(0)
    sides = {resolve_episode_spawn("random", rng=rng) for _ in range(20)}
    assert sides == {"sw", "ne"}


def test_a_short_spawn_patch_pins_sw_ne():
    if not A_SHORT.exists():
        pytest.skip("fase2_a_short.oramap missing")
    raw = A_SHORT.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        yaml0 = z.read("map.yaml").decode("utf-8")
    cells = list_mpspawn_cells(yaml0)
    assert len(cells) >= 2
    labels = classify_spawn_indices(cells)
    assert labels["sw"] != labels["ne"]
    assert cells[labels["sw"] - 1][0] < cells[labels["ne"] - 1][0]

    patched, meta = patch_oramap_bytes(raw, "sw")
    assert meta["agent_side"] == "sw"
    assert meta["enemy_cell"] == cells[labels["ne"] - 1]
    with zipfile.ZipFile(io.BytesIO(patched)) as z:
        y = z.read("map.yaml").decode("utf-8")
    block = y.split("PlayerReference@Multi1:\n", 1)[1].split("PlayerReference@", 1)[0]
    assert "LockSpawn: True" in block
    assert ("Spawn: %s" % labels["sw"]) in block

    patched_ne, meta_ne = patch_oramap_bytes(raw, "ne")
    assert meta_ne["agent_side"] == "ne"
    assert meta_ne["enemy_cell"] == cells[labels["sw"] - 1]


def test_live_module_exports_helpers():
    from rl import play_vs_checkpoint_live as m
    assert m.normalize_enemy_faction("france") == "france"
    assert m.normalize_spawn("sw") == "sw"
