# 19 — Run 45: Informe-3 surgical + reseed seed-best80

**2026-09-04.** Tras Run 44 plateau (peak WR 24.6%@29 → ~16.6%@80, medium 0/110).

## Seed
- Archivado: `rl/ckpts/Run 44 (a_short pack-total-lr 1-80)/`
- `latest.pt` = `best.pt` = `seed-best80.pt` con `iteration=0`

## Cambios Informe-3 (diff sin commit)
1. **Reward** (`eradicate_v4`): `w_timeout` 1→**6**; garrison solo si `tick <= 15000`.
2. **Harvest mask**: si hay harvs pero ninguna `is_idle`, `harvest` ilegal.
3. **Lago**: snap a flanco N/S más ancho si target es agua; `army_attack_move` stage via flanco si el segmento army→target cruza lago.
4. **PPO**: `--epochs 1`, `--clip-eps 0.15`, `--max-grad-norm 1.0` (TRAIN_ARGS + CLI).

Sigue: PACK_ARMY total>=12, LR 1e-4, `--no-war-nudge`, PFSP `medium,rl`.

## Lanzar
```
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
.\.env\Scripts\python.exe rl\auto_train.py
```
Esperá en log: epochs/clip via train args; `WAR NUDGE OFF`.
