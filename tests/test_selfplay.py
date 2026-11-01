from pathlib import Path

import numpy as np
import pytest

from ra_rl.selfplay import elo_update, plan_hybrid_jobs, schedule_kl_coefs


def test_elo_update_zero_sum_and_upset_scaling():
    r_a, r_b = elo_update(1200.0, 1200.0, score_a=1.0, k=16.0)
    assert pytest.approx(r_a, abs=1e-6) == 1208.0
    assert pytest.approx(r_b, abs=1e-6) == 1192.0
    assert pytest.approx(r_a + r_b, abs=1e-6) == 2400.0

    # Beating a stronger opponent gains more than beating an equal opponent
    upset_a, _ = elo_update(1200.0, 1400.0, score_a=1.0, k=16.0)
    assert upset_a - 1200.0 > 12.0


def test_schedule_kl_coefs_cosine_decay():
    p1, a1 = schedule_kl_coefs(1, 80, kl_prod_init=0.04, kl_prod_min=0.005, kl_army_init=0.04, kl_army_min=0.002)
    pmid, amid = schedule_kl_coefs(40, 80, kl_prod_init=0.04, kl_prod_min=0.005, kl_army_init=0.04, kl_army_min=0.002)
    pend, aend = schedule_kl_coefs(80, 80, kl_prod_init=0.04, kl_prod_min=0.005, kl_army_init=0.04, kl_army_min=0.002)

    assert p1 == 0.04 and a1 == 0.04
    assert pend == 0.005 and aend == 0.002
    assert p1 > pmid > pend
    assert a1 > amid > aend
    assert amid < pmid  # tactical KL releases faster than production KL


def test_plan_hybrid_jobs_allocates_pools_and_swaps_slots(tmp_path):
    snaps = {
        "snap_0000_init": tmp_path / "snap_0000_init.pt",
        "snap_0008": tmp_path / "snap_0008.pt",
    }
    snap_wrs = {"snap_0000_init": 0.85, "snap_0008": 0.45}
    bot_wrs = {"hard": 0.70, "brutal": 0.40, "turtle": 0.85}

    jobs = plan_hybrid_jobs(
        games=100,
        seed_base=50000,
        pool_shares=(0.30, 0.20, 0.35, 0.15),
        bots=["hard", "brutal", "turtle"],
        bot_ema_wrs=bot_wrs,
        snapshots=snaps,
        snap_ema_wrs=snap_wrs,
        anchor_spec="net:runs/v4/bc_v4/model.pt",
        maps=["singles.oramap"],
        rng=np.random.default_rng(7),
    )

    assert len(jobs) == 100
    pools = {j.pool for j in jobs}
    assert pools == {"mirror", "snapshot", "bots", "anchor"}

    # Snapshot + anchor jobs alternate Multi1 and Multi0 slots to cancel spawn asymmetry
    vs_pol = [j for j in jobs if j.pool in ("snapshot", "anchor")]
    assert any(j.learner_is_p1 for j in vs_pol)
    assert any(not j.learner_is_p1 for j in vs_pol)

    # Empty snapshot fallback folds cleanly into mirror without error
    jobs_no_snap = plan_hybrid_jobs(
        games=24,
        seed_base=60000,
        pool_shares=(0.30, 0.20, 0.35, 0.15),
        bots=["hard", "brutal", "turtle"],
        bot_ema_wrs=bot_wrs,
        snapshots={},
        snap_ema_wrs={},
        anchor_spec="scripted",
        maps=["singles.oramap"],
        rng=np.random.default_rng(11),
    )
    assert len(jobs_no_snap) == 24
    assert all(j.pool != "snapshot" for j in jobs_no_snap)
