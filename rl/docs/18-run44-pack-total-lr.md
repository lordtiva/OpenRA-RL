# 18 — Run 44: PACK_ARMY total + LR 1e-4

**2026-09-03.** Tras Run 43 (no-war-nudge): ejércitos de 200+ se clavaban mid-map porque rmy_attack_move exigía 12 en **casa**.

## Cambios
- 
_combat_total(obs) >= 12 habilita rmy_attack_move (casa o campo). Drip-4 en casa sigue bloqueado.
- LR 1.5e-4 → 1.0e-4 (spikes de clip/KL).
- **Sin** mask naval (lago Singles es legal; malo ≠ inválido).
- **Sin** cambio de clip_eps ni reward de proximidad.
- Sigue: --auto-support --no-war-nudge, PFSP medium,rl, ancla easy 50%.

## Seed
- Archivado: 
l/ckpts/Run 43 (a_short no-war-nudge 1-122)/
- latest.pt = est.pt = pesos Run43 best@80, iteration=0
- Copia: seed-best80.pt (iteration=80)

## Lanzar
\cd C:\Users\lordc\Desktop\OpenRA-RL
\=""
.\.env\Scripts\python.exe rl\auto_train.py
\Log: WAR NUDGE OFF, challengers=['medium', 'rl'], lr 1e-4.
