import numpy as np
import pytest


def compute_harmonic_score(wr_h: float, wr_b: float) -> float:
    return round(2.0 * (wr_h * wr_b) / (wr_h + wr_b + 1e-8), 3)


def test_harmonic_score():
    assert compute_harmonic_score(0.7, 0.7) == 0.7
    assert compute_harmonic_score(0.8, 0.4) == pytest.approx(0.533, abs=1e-3)
    # If one bot is 0, score is 0
    assert compute_harmonic_score(1.0, 0.0) == 0.0


def test_league_opponent_assignment():
    opponents = ["hard", "brutal", "turtle"]
    weights = [0.5, 0.3, 0.2]
    games = 100

    probs = np.array(weights, dtype=np.float64)
    probs /= probs.sum()
    counts = np.random.multinomial(games, probs)
    bot_assignments = []
    for bot_name, count in zip(opponents, counts):
        bot_assignments.extend([bot_name] * count)

    assert len(bot_assignments) == games
    assert set(bot_assignments) <= set(opponents)
