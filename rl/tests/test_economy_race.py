# Test E2E del muestreo de carrera económica: juega UNA partida corta
# con red aleatoria y verifica que el outcome traiga resumen + series.
import asyncio
import json
import sys

sys.path.insert(0, ".")

import torch

from openra_env.client import OpenRAEnv
from rl.network import AlphaLiteNet
from rl.action_adapter import Vocab
from rl.rollout import collect_one_episode


async def amain(url: str) -> int:
    net = AlphaLiteNet()
    net.eval()
    vocab = Vocab()
    async with OpenRAEnv(base_url=url, message_timeout_s=300.0) as env:
        traj, out = await collect_one_episode(
            env, net, vocab, "cpu", k_skip=8,
            temperature=1.0, max_steps=30,
            telemetry=[], macro_ticks=160)
    print(f"resultado: {out['result']} | ticks {out['ticks']} | "
          f"decisiones {out['decisions']} | reward {out['episode_reward']}")
    race = out.get("economy_race")
    print("economy_race:", json.dumps(race))
    series = out.pop("economy_race_series")
    print(f"series: {len(series['ticks'])} muestras | "
          f"primeros ticks: {series['ticks'][:8]}")
    ok = bool(race) and len(series["ticks"]) >= 3 \
        and len(series["own_wealth"]) == len(series["ticks"])
    print("E2E-OK" if ok else "E2E-FALLO")
    return 0 if ok else 1


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8011"
    raise SystemExit(asyncio.run(amain(url)))
