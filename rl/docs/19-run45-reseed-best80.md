# 19 — Run 45: reseed seed-best80 (post Run 44)

**2026-09-04.** Run 44 (PACK_ARMY total + LR 1e-4) plateaued: peak WR **24.6% @29**, end **~16.6% @80**, medium **0/110**. See `Informe-3.md`.

## Seed
- Archivado: `rl/ckpts/Run 44 (a_short pack-total-lr 1-80)/`
- `latest.pt` = `best.pt` = pesos de `seed-best80.pt` con `iteration=0`
- `seed-best80.pt` intacto (iteration=80)

## Config al lanzar (igual Run 44 hasta aplicar Informe-3)
- `--auto-support --no-war-nudge`
- PFSP challengers `medium,rl`, ancla easy 50%
- PACK_ARMY total >= 12, LR `1.0e-4`
- **Pendiente** (Informe-3, no aplicado aún): timeout↑, garrison cap post-15k, mask harvest idle, epochs 1 / clip_eps 0.15

## Lanzar
```
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
.\.env\Scripts\python.exe rl\auto_train.py
```
