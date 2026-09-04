# 21 — Run 47: Map QSA (block top-k)

**2026-09-04.** Run 46 smoke (burn-in 8 + entity XF top-k 16): **era WR 42.5%**, **wr20 45%** @20 iters.

## QSA de mapa (nuevo)
- Bloques `8×8`, **top-8** bloques para la cabeza de celda.
- Score de bloque = media de logits densos (prior) + indexer aprendido (`qsa_query` · `qsa_key` sobre fmap).
- Celdas fuera de los top-k → `-1e4`. STE para gradiente del indexer.
- **No** IndexShare con `cell_head` (keys aparte, init chico).
- Flags: `--qsa-topk 8 --qsa-block 8`

Sigue activo: burn-in 8, xf-topk 16, Informe-3, PACK total, LR 1e-4, no-war-nudge, PFSP.

## Seed
- Archivado: `rl/ckpts/Run 46 (a_short burnin-topk smoke 1-20)/`
- `latest`/`best` = Run46 **latest@20** (no best@1) con `iteration=0`

## Lanzar (smoke 20)
```
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
.\.env\Scripts\python.exe rl\auto_train.py
```
Log: `map QSA top-k=8 block=8` + `entity XF top-k=16`.
