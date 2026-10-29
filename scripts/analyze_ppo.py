import json
from pathlib import Path

path = Path("runs/phase3/ppo_v1/metrics.jsonl")
lines = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n") if line.strip()]

print(f"Total iterations logged: {len(lines)}")

evals = [m for m in lines if "eval" in m]
print("\n--- EVALUACIONES OFICIALES (100 partidas vs hard, semillas 20000-20099) ---")
for e in evals:
    it = e["iteration"]
    ev = e["eval"]
    ci = ev["win_rate_ci95"]
    outcomes = ev["outcomes"]
    nb = "[BEST]" if e.get("new_best") else ""
    print(f"Iter {it:3d}: Win Rate: {ev['win_rate']*100:5.1f}% | CI95: [{ci[0]*100:4.1f}%, {ci[1]*100:4.1f}%] | W:{outcomes.get('win',0):2d} L:{outcomes.get('lose',0):2d} T:{outcomes.get('truncated',0):2d} | {nb}")

rollout_wrs = [m["win_rate"] for m in lines]
print("\n--- ROLLOUT SUMMARY (48 partidas por iter vs hard) ---")
print(f"Rollout Win Rate: Inicial={rollout_wrs[0]*100:.1f}%, Min={min(rollout_wrs)*100:.1f}%, Max={max(rollout_wrs)*100:.1f}%, Final={rollout_wrs[-1]*100:.1f}%")

print("\n--- PROGRESIÓN POR BLOQUES DE 10 ITERACIONES ---")
for chunk_start in range(0, len(lines), 10):
    chunk = lines[chunk_start:chunk_start+10]
    avg_wr = sum(m["win_rate"] for m in chunk) / len(chunk)
    avg_ticks = sum(m["mean_ticks"] for m in chunk) / len(chunk)
    avg_dec = sum(m["mean_decisions"] for m in chunk) / len(chunk)
    avg_pi_prod = sum(m["pi_loss_prod"] for m in chunk) / len(chunk)
    avg_pi_army = sum(m["pi_loss_army"] for m in chunk) / len(chunk)
    avg_v_prod = sum(m["val_loss_prod"] for m in chunk) / len(chunk)
    avg_v_army = sum(m["val_loss_army"] for m in chunk) / len(chunk)
    avg_ent_p = sum(m["ent_prod"] for m in chunk) / len(chunk)
    avg_ent_a = sum(m["ent_army"] for m in chunk) / len(chunk)
    avg_kl_p = sum(m["kl_prod"] for m in chunk) / len(chunk)
    avg_kl_a = sum(m["kl_army"] for m in chunk) / len(chunk)
    total_time = sum(m["it_s"] for m in chunk)
    early_stops = sum(1 for m in chunk if m.get("early_stopped"))
    print(f"Iters {chunk_start+1:2d}-{chunk_start+len(chunk):3d} | WR: {avg_wr*100:5.1f}% | Ticks: {avg_ticks:5.0f} | Decs: {avg_dec:4.1f} | PiLoss (P/A): {avg_pi_prod:+.3f}/{avg_pi_army:+.3f} | VLoss (P/A): {avg_v_prod:.3f}/{avg_v_army:.3f} | Ent (P/A): {avg_ent_p:.3f}/{avg_ent_a:.3f} | KL: {avg_kl_p:.3f}/{avg_kl_a:.3f} | ES: {early_stops} | T: {total_time/60:.1f}m")

# Checkpoint best details
best_ckpt = Path("runs/phase3/ppo_v1/checkpoint_best.pt")
if best_ckpt.exists():
    import torch
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    print("\n--- CHECKPOINT_BEST.PT ---")
    print(f"Iteration: {ckpt.get('iteration')}")
    print(f"Eval stats: {ckpt.get('eval')}")
