# -*- coding: utf-8 -*-
"""Live/viewer lobby helpers: enemy faction + spawn (SW/NE) without touching train defaults.

Spawn pin works by patching the scenario .oramap PlayerReference LockSpawn/Spawn
before CreateSession. C# SyncClientToPlayerReference already honors those locks.
Agent stays on Multi1 (FastAdvance primary); bot on Multi0. SW/NE are geometry
labels over mpspawn cells (a_short: spawn1~(12,16) SW, spawn2~(95,11) NE).
"""
from __future__ import annotations

import io
import random
import re
import zipfile
from typing import Iterable

from rl.allies import (
    ALLIED_COUNTRIES,
    DEFAULT_ENEMY_FACTION,
    DEFAULT_PLAYER_FACTION,
    SOVIET_COUNTRIES,
    resolve_enemy_faction,
    resolve_player_faction,
)

# Live agent must stay Allies (RandomAllies or a country).
PLAYER_FACTION_CHOICES = (DEFAULT_PLAYER_FACTION,) + ALLIED_COUNTRIES
ENEMY_FACTION_CHOICES = (
    DEFAULT_ENEMY_FACTION,
    "RandomAllies",
    "RandomSoviet",
) + ALLIED_COUNTRIES + SOVIET_COUNTRIES
SPAWN_CHOICES = ("random", "sw", "ne")

_MPSPAWN_RE = re.compile(
    r"(?P<indent>\t*)Actor\d+:\s*mpspawn\s*\r?\n"
    r"(?:(?P=indent)\t[^\n]*\r?\n)*?"
    r"(?P=indent)\tLocation:\s*(?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)",
    re.MULTILINE,
)
_PLAYER_BLOCK_RE = re.compile(
    r"(?P<header>\tPlayerReference@(?P<slot>Multi[01]):\r?\n)"
    r"(?P<body>(?:\t\t[^\n]*\r?\n)*)",
    re.MULTILINE,
)


def _norm_yaml(map_yaml: str) -> str:
    return map_yaml.replace("\r\n", "\n").replace("\r", "\n")


def normalize_spawn(raw: str | None) -> str:
    s = str(raw or "random").strip().lower()
    aliases = {
        "r": "random",
        "rnd": "random",
        "auto": "random",
        "0": "random",
        "sw": "sw",
        "southwest": "sw",
        "south-west": "sw",
        "1": "sw",
        "ne": "ne",
        "northeast": "ne",
        "north-east": "ne",
        "2": "ne",
    }
    out = aliases.get(s, s)
    if out not in SPAWN_CHOICES:
        raise ValueError(f"spawn must be one of {SPAWN_CHOICES} (got {raw!r})")
    return out


def normalize_player_faction(raw: str | None) -> str:
    resolved = resolve_player_faction(raw)
    low = resolved.lower()
    allowed = {c.lower() for c in PLAYER_FACTION_CHOICES}
    if low not in allowed:
        raise ValueError(
            "player faction must be Allies ("
            + ", ".join(PLAYER_FACTION_CHOICES)
            + f"); got {raw!r} -> {resolved!r}"
        )
    for c in PLAYER_FACTION_CHOICES:
        if c.lower() == low:
            return c
    return DEFAULT_PLAYER_FACTION


def normalize_enemy_faction(raw: str | None) -> str:
    resolved = resolve_enemy_faction(raw)
    low = resolved.lower()
    allowed = {c.lower() for c in ENEMY_FACTION_CHOICES}
    if low not in allowed:
        raise ValueError(
            f"enemy faction must be one of {ENEMY_FACTION_CHOICES}; got {raw!r}"
        )
    for c in ENEMY_FACTION_CHOICES:
        if c.lower() == low:
            return c
    return DEFAULT_ENEMY_FACTION


def list_mpspawn_cells(map_yaml: str) -> list[tuple[int, int]]:
    """mpspawn cells in map.yaml order (= OpenRA spawn indices 1..N)."""
    map_yaml = _norm_yaml(map_yaml)
    return [(int(m.group("x")), int(m.group("y"))) for m in _MPSPAWN_RE.finditer(map_yaml)]


def classify_spawn_indices(cells: Iterable[tuple[int, int]]) -> dict[str, int]:
    """Map geometry labels -> 1-based spawn index. Needs >=2 mpspawns."""
    cells = list(cells)
    if len(cells) < 2:
        raise ValueError(f"need >=2 mpspawn cells to pin SW/NE (got {len(cells)})")
    ranked = sorted(enumerate(cells, start=1), key=lambda iv: iv[1][0] - iv[1][1])
    sw_i, _ = ranked[0]
    ne_i, _ = ranked[-1]
    if sw_i == ne_i:
        raise ValueError("could not separate SW/NE spawn cells")
    return {"sw": sw_i, "ne": ne_i}


def resolve_episode_spawn(asked: str | None, rng: random.Random | None = None) -> str:
    """Return concrete sw|ne for this episode.

    --spawn random => pick sw/ne each episode (real rotation).
    """
    s = normalize_spawn(asked)
    if s == "random":
        r = rng or random
        return r.choice(("sw", "ne"))
    return s


def _upsert_player_spawn(body: str, spawn_index: int) -> str:
    body = _norm_yaml(body)
    lines = body.splitlines(keepends=True)
    out: list[str] = []
    saw_spawn = saw_lock = False
    for line in lines:
        if re.match(r"\t\tSpawn:\s*", line):
            out.append(f"\t\tSpawn: {spawn_index}\n")
            saw_spawn = True
            continue
        if re.match(r"\t\tLockSpawn:\s*", line):
            out.append("\t\tLockSpawn: True\n")
            saw_lock = True
            continue
        out.append(line if line.endswith("\n") else line + "\n")
    if not saw_lock:
        out.insert(0, "\t\tLockSpawn: True\n")
    if not saw_spawn:
        insert_at = 1 if out and out[0].startswith("\t\tLockSpawn:") else 0
        out.insert(insert_at, f"\t\tSpawn: {spawn_index}\n")
    return "".join(out)


def patch_map_yaml_spawns(
    map_yaml: str,
    agent_side: str,
    agent_slot: str = "Multi1",
    enemy_slot: str = "Multi0",
) -> tuple[str, dict]:
    """Lock Multi slots to SW/NE. agent_side is sw|ne."""
    map_yaml = _norm_yaml(map_yaml)
    side = normalize_spawn(agent_side)
    if side == "random":
        raise ValueError("patch_map_yaml_spawns needs concrete sw|ne")
    cells = list_mpspawn_cells(map_yaml)
    labels = classify_spawn_indices(cells)
    agent_idx = labels[side]
    enemy_side = "ne" if side == "sw" else "sw"
    enemy_idx = labels[enemy_side]
    want = {agent_slot: agent_idx, enemy_slot: enemy_idx}

    def repl(m: re.Match) -> str:
        slot = m.group("slot")
        if slot not in want:
            return m.group(0)
        return m.group("header").replace("\r\n", "\n") + _upsert_player_spawn(
            m.group("body"), want[slot]
        )

    patched = _PLAYER_BLOCK_RE.sub(repl, map_yaml)
    found = {m.group("slot") for m in _PLAYER_BLOCK_RE.finditer(map_yaml)}
    missing = [s for s in want if s not in found]
    if missing:
        raise ValueError(f"map.yaml missing PlayerReference for {missing}")
    meta = {
        "agent_side": side,
        "enemy_side": enemy_side,
        "agent_spawn_index": agent_idx,
        "enemy_spawn_index": enemy_idx,
        "agent_cell": cells[agent_idx - 1],
        "enemy_cell": cells[enemy_idx - 1],
        "spawns": cells,
    }
    return patched, meta


def patch_oramap_bytes(
    oramap: bytes,
    agent_side: str,
    agent_slot: str = "Multi1",
    enemy_slot: str = "Multi0",
) -> tuple[bytes, dict]:
    """Return new .oramap bytes with LockSpawn pins + meta (enemy_cell for beacon)."""
    in_buf = io.BytesIO(oramap)
    out_buf = io.BytesIO()
    meta: dict = {}
    with zipfile.ZipFile(in_buf, "r") as zin, zipfile.ZipFile(
        out_buf, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        names = zin.namelist()
        if "map.yaml" not in names:
            raise ValueError("oramap missing map.yaml")
        for name in names:
            data = zin.read(name)
            if name == "map.yaml":
                text = data.decode("utf-8")
                text, meta = patch_map_yaml_spawns(
                    text, agent_side, agent_slot=agent_slot, enemy_slot=enemy_slot
                )
                data = text.encode("utf-8")
            zout.writestr(name, data)
    return out_buf.getvalue(), meta
