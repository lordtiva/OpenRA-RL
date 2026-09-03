"""Helpers for RL-vs-RL peer observations / frozen opponent acts."""
from __future__ import annotations

from openra_env.models import (
    BuildingInfoModel,
    EconomyInfo,
    MapInfoModel,
    MilitaryInfo,
    OpenRAObservation,
    ProductionInfoModel,
    UnitInfoModel,
)


def peer_obs_from_metadata(obs: OpenRAObservation) -> OpenRAObservation | None:
    """Rebuild Multi0 OpenRAObservation from observation.metadata['peer']."""
    peer = getattr(obs, "peer", None)
    if not isinstance(peer, dict):
        meta = getattr(obs, "metadata", None) or {}
        peer = meta.get("peer")
    if not isinstance(peer, dict):
        return None
    return OpenRAObservation(
        tick=peer.get("tick", 0),
        economy=EconomyInfo(**peer.get("economy", {})),
        military=MilitaryInfo(**peer.get("military", {})),
        units=[UnitInfoModel(**u) for u in peer.get("units", [])],
        buildings=[BuildingInfoModel(**b) for b in peer.get("buildings", [])],
        production=[ProductionInfoModel(**p) for p in peer.get("production", [])],
        visible_enemies=[UnitInfoModel(**u) for u in peer.get("visible_enemies", [])],
        visible_enemy_buildings=[
            BuildingInfoModel(**b) for b in peer.get("visible_enemy_buildings", [])
        ],
        map_info=MapInfoModel(**peer.get("map_info", {})),
        available_production=peer.get("available_production", []),
        done=peer.get("done", False),
        reward=peer.get("reward"),
        result=peer.get("result", ""),
        spatial_map=peer.get("spatial_map", ""),
        spatial_channels=peer.get("spatial_channels", 0),
    )
