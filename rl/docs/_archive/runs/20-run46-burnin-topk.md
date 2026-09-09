# 20 — Run 46: Burn-in 8 + entity XF top-k 16

**2026-09-04.** Tras Run 45 (Informe-3, iters 1–40): harvest/timeout OK, pero clip 0.45–0.63 mid-run y grad 14–17.

## Cambios
1. **Burn-in 8** (`--burn-in 8`): cada segmento BPTT antepone hasta 8 pasos `_burn` que avanzan la GRU sin loss (R2D2).
2. **Top-k=16** en `unit_xf` (`--xf-topk 16`): atención sparse sobre tokens de entidades; mismos pesos MHA (state_dict compatible). No es el QSA de *mapa* del doc 13 — es el filtro de entidades acordado.

Sigue: Informe-3 (timeout 6, garrison ≤15k, harvest idle mask, epochs 1, clip 0.15, grad clip 1.0), PACK total≥12, LR 1e-4, no-war-nudge, PFSP medium+rl.

## Seed
- Archivado: `rl/ckpts/Run 45 (a_short informe3 1-40)/`
- `latest`/`best` = best@25 de Run45 con `iteration=0` (+ `seed-best25-run45.pt`)

## Lanzar
```
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
.\.env\Scripts\python.exe rl\auto_train.py
```
Log: `entity XF top-k=16`, burn-in vía TRAIN_ARGS. Medir clip/v_loss/grad vs Run 45.
