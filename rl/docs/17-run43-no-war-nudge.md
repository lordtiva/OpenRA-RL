# 17 — Run 43: PPO dueño de la guerra (war nudge off)

**2026-09-03.** Experimento: seed = Run 42 est@79, métricas desde iter 1, **sin war nudge**.

## Qué cambió
- Flag nuevo: \--no-war-nudge\ (con \--auto-support\). Apaga raid / push / fog-scout. Siguen repair, power_down, harvest idle, stance, deploy MCV, sell wrecks, auto-proc/tent.
- Liga: 50% easy ancla, 50% PFSP \medium,rl\ (sin beginner, easy no se doble-cuenta).
- Archivado: l/ckpts/Run 42 (a_short pfsp-rl 1-120)/\.
- Seed: \latest.pt\ = \est.pt\ = pesos de iter 79, \iteration=0\. Copia congelada: \seed-best79.pt\.
- PFSP stats reseteadas (liga nueva). \prev20.pt\ no está en root.

## Lanzar
\cd C:\Users\lordc\Desktop\OpenRA-RL
\=""
.\.env\Scripts\python.exe rl\auto_train.py
\En el log: \AUTO-SUPPORT ON, WAR NUDGE OFF\ y \challengers=['medium', 'rl']\.
