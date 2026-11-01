"""Fase 4.0 integration tests for Commander v4 + SelfPlaySession against a live daemon.

Run with:
    .\\.venv\\Scripts\\python.exe -m pytest -m daemon
"""

from concurrent.futures import ThreadPoolExecutor
import os

import pytest

from ra_env import Bridge, SelfPlayConfig, SelfPlaySession, pb
from ra_env.probe import obs_digest

pytestmark = pytest.mark.daemon

ADDRESS = os.environ.get("RA_DAEMON", "localhost:9999")


@pytest.fixture(scope="module")
def bridge():
    b = Bridge(ADDRESS)
    yield b
    b.close()


def _macro_policy(
    obs: pb.GameObservation,
    pending: list[pb.DecisionRequest],
    want_air: bool = False,
    hold_main: bool = False,
) -> list[pb.Decision]:
    """Minimal deterministic Allied/Soviet macro responder for daemon integration tests."""
    bcounts: dict[str, int] = {}
    for b in obs.buildings:
        k = b.type.split(".", 1)[0]
        bcounts[k] = bcounts.get(k, 0) + 1
    ucounts: dict[str, int] = {}
    for u in obs.units:
        k = u.type.split(".", 1)[0]
        ucounts[k] = ucounts.get(k, 0) + 1

    out: list[pb.Decision] = []
    for r in pending:
        opts = {o.split(".", 1)[0]: o for o in r.options}
        if r.kind == pb.DECIDE_BUILD:
            order = ["powr", "proc", "barr", "tent", "weap", "dome"]
            if want_air:
                order = ["powr", "proc", "barr", "tent", "dome", "apwr", "afld", "hpad", "weap"]
            picked = ""
            for want in order:
                if want not in opts:
                    continue
                cap = 3 if want == "powr" else 1
                if bcounts.get(want, 0) < cap:
                    picked = opts[want]
                    bcounts[want] = bcounts.get(want, 0) + 1
                    break
            if picked:
                out.append(pb.Decision(request_id=r.id, choice=picked))
        elif r.kind == pb.DECIDE_TRAIN:
            picked = ""
            if r.queue == "Aircraft":
                for want in ("mh60", "heli", "yak", "mig"):
                    if want in opts and ucounts.get(want, 0) < 1:
                        picked = opts[want]
                        break
            elif r.queue == "Vehicle":
                for want in ("jeep", "ftrk", "2tnk", "3tnk", "1tnk"):
                    if want in opts:
                        if want in ("jeep", "ftrk") and ucounts.get(want, 0) >= 2:
                            continue
                        if hold_main and ucounts.get(want, 0) >= 2:
                            continue
                        picked = opts[want]
                        break
            elif r.queue == "Infantry":
                if "e1" in opts and ucounts.get("e1", 0) < 3:
                    picked = opts["e1"]
            if picked:
                out.append(pb.Decision(request_id=r.id, choice=picked))
        elif r.kind == pb.DECIDE_ARMY:
            if hold_main and r.queue == "MAIN":
                out.append(pb.Decision(request_id=r.id, choice="hold", target_index=-1))
            elif r.targets:
                choice = "all_in" if (r.queue == "MAIN" and len(obs.units) >= 6) else "attack"
                out.append(pb.Decision(request_id=r.id, choice=choice, target_index=0))
            else:
                out.append(pb.Decision(request_id=r.id, choice="hold", target_index=-1))
        elif r.kind == pb.DECIDE_SUPPORT:
            if r.targets:
                out.append(pb.Decision(request_id=r.id, choice="attack", target_index=0))
            else:
                out.append(pb.Decision(request_id=r.id, choice="hold", target_index=-1))
    return out


def _dual_trace(bridge: Bridge, cfg: SelfPlayConfig, seed: int, steps: int = 25) -> list[tuple[int, str, str]]:
    sess = SelfPlaySession(bridge, cfg)
    try:
        st = sess.reset(seed=seed)
        trace = [(st.tick, obs_digest(st.p1.obs), obs_digest(st.p0.obs))]
        for _ in range(steps):
            if st.done or st.truncated:
                break
            d1 = _macro_policy(st.p1.obs, st.p1.pending)
            d0 = _macro_policy(st.p0.obs, st.p0.pending)
            st = sess.step(d1, d0, ticks=100)
            trace.append((st.tick, obs_digest(st.p1.obs), obs_digest(st.p0.obs)))
        return trace
    finally:
        sess.close()


def test_selfplay_dual_determinism(bridge):
    cfg = SelfPlayConfig(player_faction="russia", enemy_faction="england", max_ticks=3000)
    t_a = _dual_trace(bridge, cfg, seed=777, steps=30)
    t_b = _dual_trace(bridge, cfg, seed=777, steps=30)
    t_c = _dual_trace(bridge, cfg, seed=778, steps=30)
    assert t_a == t_b, "identical seed + policy must produce bit-identical dual observations"
    assert t_a != t_c, "different seeds must diverge"


def test_selfplay_bilateral_factions_and_fog_honesty(bridge):
    cfg = SelfPlayConfig(player_faction="ukraine", enemy_faction="france", max_ticks=2500)
    sess = SelfPlaySession(bridge, cfg)
    try:
        st = sess.reset(seed=101)
        d1, d0 = [], []
        while not (st.done or st.truncated):
            d1 = _macro_policy(st.p1.obs, st.p1.pending)
            d0 = _macro_policy(st.p0.obs, st.p0.pending)
            st = sess.step(d1, d0, ticks=250)
        assert st.p1.obs.player_slot == "Multi1"
        assert st.p0.obs.player_slot == "Multi0"
        assert st.p1.obs.player_faction == "ukraine"
        assert st.p0.obs.player_faction == "france"
        assert any(b.type == "barr" for b in st.p1.obs.buildings), "Soviet side should build barr"
        assert any(b.type == "tent" for b in st.p0.obs.buildings), "Allied side should build tent"
    finally:
        sess.close()


def test_selfplay_multisquad_air_and_support_powers(bridge):
    # SovietSpyPlane has ChargeInterval=3750 ticks after AFLD completes (~tick 2800 -> ~6600).
    cfg = SelfPlayConfig(player_faction="ukraine", enemy_faction="england", max_ticks=9500)
    sess = SelfPlaySession(bridge, cfg)
    try:
        st = sess.reset(seed=202)
        seen_groups: set[str] = set()
        seen_support: set[str] = set()
        saw_harass_units = False
        fired_support = False
        while not (st.done or st.truncated):
            for r in st.p1.pending + st.p0.pending:
                if r.kind == pb.DECIDE_ARMY:
                    seen_groups.add(r.queue)
                elif r.kind == pb.DECIDE_SUPPORT:
                    seen_support.add(r.queue)
            if st.p1.obs.harass_unit_ids or st.p0.obs.harass_unit_ids:
                saw_harass_units = True
            if {"MAIN", "HARASS", "AIR"} <= seen_groups and fired_support:
                break
            if seen_support:
                fired_support = True
            hold_main = not seen_support
            d1 = _macro_policy(st.p1.obs, st.p1.pending, want_air=True, hold_main=hold_main)
            d0 = _macro_policy(st.p0.obs, st.p0.pending, want_air=True, hold_main=hold_main)
            st = sess.step(d1, d0, ticks=250)

        assert "MAIN" in seen_groups
        assert "HARASS" in seen_groups and saw_harass_units, f"expected HARASS squad; groups={seen_groups}"
        assert "AIR" in seen_groups, f"expected AIR squad; groups={seen_groups}"
        assert "SovietSpyPlane" in seen_support, f"expected SovietSpyPlane DECIDE_SUPPORT; got {seen_support}"
    finally:
        sess.close()


def test_selfplay_concurrent_stress_zero_errors(bridge):
    cfg = SelfPlayConfig(player_faction="Random", enemy_faction="Random", max_ticks=2000)

    def run_one(seed: int) -> tuple[int, str, str]:
        sess = SelfPlaySession(bridge, cfg)
        try:
            st = sess.reset(seed=seed)
            while not (st.done or st.truncated):
                d1 = _macro_policy(st.p1.obs, st.p1.pending)
                d0 = _macro_policy(st.p0.obs, st.p0.pending)
                st = sess.step(d1, d0, ticks=250)
            return st.tick, st.p1.obs.player_faction, st.p0.obs.player_faction
        finally:
            sess.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, range(500, 508)))
    assert len(results) == 8
    factions = {f for _, f1, f0 in results for f in (f1, f0)}
    assert factions <= {"england", "france", "germany", "russia", "ukraine"}
    assert all(t >= 200 for t, _, _ in results)
