# Capa 2c — Identidad de entidad y matchup (spec de implementación)

> **Fecha:** 2026-08-31 · **No reemplaza** `12-plan-4-capas-siguiente-nivel.md` ni `13-capa0-status-post-run8.md`.
> **Qué es:** deuda de Capa 2 que el corte 1010 no cerró: el transformer ve 48 propias anónimas; Ch8 es densidad; `attack` pega al más cercano. Este doc es el PR plan para implementarlo **sin tabula rasa**.
> **No es Capa 3.** Capa 3 = sparring (`easy` → `hard` → RL-vs-RL). 2c es el sesgo inductivo para que ese sparring *pueda* enseñar “e3 vs tanque”.
> **No es Capa 2b.** 2b = QSA / GDN / GRU 512 (mapa / memoria). Otro corte, después.

---

## Por qué ahora

Capa 2 del 12 pedía scatter de **equipo + HP + rol**. Lo shipped (corte 1010) es transformer 2×4h d=64 + scatter de 10 floats propios + `celda|unidad`. El `type` (`e1`, `e3`, `1tnk`) **llega en el proto y se tira**. `visible_enemies[].type` también. El AutoTarget del engine aplica arma×armadura; la política no.

Operador (iter **980**, régimen APM 50 ticks / 1000 steps, resume 970, vs **easy**): **wr global (`wins/total` del log) ~30%**. Eso es el acumulado de era, no wr20. Run 21 (80 t, 952–978) era 8.8% win / `defense_loss`≈−4. 30% no promociona `hard` ni self-play. Sí justifica arrancar 2c: easy ya no es 0/140, y el techo visible sigue siendo “no ve el raid / no distingue roles”.

Bar para **hard** (Capa 3): wr20 vs easy decente **y** `defense_loss` que no sea −4 plano. 2c no espera a hard.

---

## Lo que hay hoy (no re-diagnosticar)

| Superficie | Estado |
|---|---|
| Slots propios | **PR-A hecho:** `MAX_UNITS=96` + `select_unit_slots` combat-first. Features siguen 10-d anónimas. |
| Features / slot | 10-d: HP, `can_attack`, idle, speed, range, XP, stance, x, y, facing. **Sin rol.** |
| Enemigos | Ch8 = conteo. Escalar = `n_enemies`. Transformer = 0 tokens rivales. |
| `attack` | Slot propio + celda → `_nearest_enemy_at_cell`. No elige actor. |
| Producción | Cabeza de ítems **sí** usa `rl.roles` (`infantry_basic`, `infantry_antiarmor`, `tank_medium`…). Puede entrenar el counter a ciegas; no ve de qué está hecho el rival. |
| Capa 2 | xf residual (scale aprendible), scatter 8 ch, `cell_head` 296-in. ~2.89M. Net2Net desde 922 ya probado. |
| Acción / APM | 1 decisión / 50 ticks (~2 s). AutoTarget cubre el micro en rango. |

El proto ya trae todo lo que falta: `UnitInfoModel.type` en `obs.units` y `obs.visible_enemies`. C# `ATTACK` ya acepta `target_actor_id`. No hay trabajo de bridge.

---

## Orden: tres PRs, un régimen por vez

No un mega-PR. No 96 + rol + enemigos + cabeza nueva el mismo resume. Cada corte: archivar, resume del **best easy** (o latest si el best es viejo beginner — `bot_type` ya lo cubre), un cambio, 20 iters de smoke.

| PR | Nombre | Qué | Pesos | Resume | Smoke (20 iters) |
|---|---|---|---|---|---|
| **A** | Set | `MAX_UNITS` 48→**96** + selección **combat-first** | **1:1** (ni un Linear nuevo) | **Hecho corte 983, resume 976** | H no colapsa; `n_unit_valid` media sube; `defense_loss` no empeora |
| **B** | Identidad | Embedding de rol + bit de equipo + tokens enemigos visibles + scatter de esas feats | Net2Net (zero-pad) | best de A | H no colapsa; `xf_scale` no explota; wr20 no a 0 |
| **C** | Pointer de ataque | Cabeza `enemy_slot` solo en `attack`; F1 usa ese `actor_id` | Cabeza nueva en 0; tronco cargado | best de B | `attack` deja de ser 100% “más cercano”; no NaN; legal mask |

Si A no mueve `defense_loss` en ~20–40 iters, **no** saltar a 128. El siguiente palanca de A es el filtro (ya va en A); si el filtro+96 sigue ciego, entonces B (rol: harv vs e1 en el set). No 256.

**No** en estos PRs: QSA, GDN, GRU 512, APC/`enter_transport`, `hard`, self-play, `SUPPORT_ASSAULT=True`, beacon Ch7/Ch8, dest-credit.

---

## PR-A — Set: 96 slots + combat-first

### Por qué 96, no 128 ni 256

Los pesos del xf / `unit_mlp` / scorer **no crecen con n**. Cómputo de atención sí: \(O(n^2)\).

| n | \(n^2\) vs 48 | Lectura |
|---|---|---|
| 48 | 1× | Hoy. Veteranos de `actor_id` bajo. |
| **96** | **4×** | 2× ejército, ckpt 1:1. El experimento. |
| 128 | ~7× | Mismo sesgo “viejos”; no vale un régimen. |
| 256 | 28× | Casi el blob de 300. Softmax a 256 e1 clones, 1 click / 2 s. PPO explora índice, no rol. |

VRAM 2070: irrelevante (el U-Net manda). El argumento es muestra y crédito.

Con 300 actores, 256 **oldest** igual tira los 44 más nuevos. El raid del easy suele ser gente **nueva**. Por eso A no es solo el techo: es **quién entra**.

### Selección (una sola función)

```
candidatos = obs.units
prioridad:
  1. combate (can_attack y no harv/mcv) a ≤18 de un visible_enemy
     o ≤18 de un edificio propio (raid en casa)
  2. resto de combate
  3. harv / mcv / otros
tomar min(96, len), ESTABLE por actor_id dentro de cada cubo
re-ordenar el tensor final por actor_id   # slot temporal estable
```

El sort final por `actor_id` es obligatorio: la cabeza 2 elige slot; si el rifle salta de índice 3→40 cada bloque, el xf no asocia.

**No** mezclar “los 96 más nuevos” (el GRU pierde veteranos). Combat-first + pad con viejos.

### Código

| Archivo | Cambio |
|---|---|
| `rl/obs_encoding.py` | `MAX_UNITS = 96`. Extraer `select_unit_slots(obs) -> list` (prioridad + sort). `unit_slots` la usa. Constante `UNIT_FEAT_DIM = 10` (hoy está mágico). |
| `rl/action_adapter.py` | Misma `select_unit_slots` para `unit_ids` (hoy re-sortea `actor_id[:48]` a mano). **Una** función. |
| `rl/rollout.py` / visor | Siguen `unit_slots`; no duplicar. |
| `rl/network.py` | Nada de pesos. `MAX_UNITS` ya se importa. Print de boot: `Capa 2: transformer 96 + combat-first`. |
| `rl/train.py` | El print de Capa 2. |

Net2Net: `load_state_dict` 1:1. No `adapt_capa2`. Adam **no** fresco (no hay keys nuevas).

### Tests

- 30 combate cerca de un enemigo + 80 e1 viejos en el ore → los 30 entran, el resto llena a 96, tensor ordenado por id.
- 10 unidades → `valid` 10, pad 86.
- Adapter y encoding eligen **los mismos** `actor_id`.
- Forward `unit_feats` `[1,96,10]` carga `best.pt` / 970 sin missing keys.

### Criterio para pasar a B

20–40 iters vs easy, H sana, sin NaN. `defense_loss` **o** wr20 mejor que el piso 980 (no hace falta 4/4). Si empeora 15 iters → restore A-resume, no empujar B encima.

---

## PR-B — Identidad: rol + enemigos (el matchup)

Este es el PR que responde “¿puede inferir esta unidad vs aquella?”. A sola no: sigue anónima.

### Observación

`UNIT_FEAT_DIM` pasa de 10 a **11** en el tensor numérico:

```
[hp, can_attack, idle, speed, range, xp, stance, x, y, facing, team]
team = 0.0 propio, 1.0 enemigo
```

Rol **no** va one-hot en el obs: va `role_id: int64 [B,U]` paralelo, vocab estable de `rl.roles` + `"misc"` (id 0 = pad). `role_of(u.type)` ya existe.

Dos set concatenados:

```
own  = select_unit_slots(obs)           # ≤96
ene  = visible_enemies fog-limited      # ≤ MAX_ENEMIES=32
      sort by actor_id
tokens = own ++ ene                     # ≤128, mask el pad
```

32 enemigos visibles alcanzan (niebla). No 96 rivales: el easy no mete 96 en pantalla y \(128^2\) ya es ~7× el 48 original.

Scatter: mismas 11-d (o cat con role-emb) sobre el fmap. Equipo queda en el canal `team` → la celda distingue blob propio vs raid. Hoy scatter solo pinta propias.

Pool del GRU: **solo tokens propios** (`team==0`). Si mezclás enemigos en el mean, el hidden 970 se corre. El xf **sí** ve los 128 (atención cruzada = el matchup).

### Red (Net2Net, patrón corte 1010)

Constantes nuevas:

```
ROLE_EMB_DIM = 8
MAX_ENEMIES = 32
UNIT_FEAT_DIM = 11          # 10 + team
UNIT_MLP_IN = 10 + 1 + 8    # feats base + team + role_emb  (=19)
```

Módulos:

- `role_emb = nn.Embedding(n_roles, 8)` — init normal chico / 0.
- `unit_mlp[0]`: `Linear(10,128)` → `Linear(19,128)`.
- `scatter_proj`: `Linear(10,8)` → `Linear(19,8)` **o** scatter de `cat(feats11, role_emb)`.
- `unit_scorer[0]`: `Linear(10+416+64, 256)` → `Linear(19+416+64, 256)` (scorer sobre el vector que ya vio el rol; alternativa: seguir 11-d crudo + hidden — peor. Preferir el token 128-d **o** cat 19-d).

Receta de pad (igual `adapt_capa2_state_dict`):

```
new_w.zero_()
new_w[:, :10] = old_w[:, :10]     # HP..facing
# cols 10: team, 11:18 role_emb → 0
```

Keys nuevas (`role_emb`) las pone `load_state_dict` missing. `xf_*`, GRU, U-Net, `head_type`, `item_*`, `value_head` 1:1.

Adam: fresco **solo** en `role_emb` + filas/cols nuevas, **o** Adam fresco global como 1010 (ya sabemos que el tronco aguanta). Preferir fresco global: un hyperparam, el 1010 no se rompió.

`unit_xf_scale` se **carga** (ya no es 0). No resetearlo: B no es otro Net2Net-a-identidad, es ampliar el token.

Máscara: `src_key_padding_mask` como ahora. Slots enemigos `unit_valid=True` pero **ilegales** en `dist_unit` (la cabeza de unidad propia no puede elegir un 2tnk rival). `unit_own_mask` vs `unit_valid`.

### Código

| Archivo | Cambio |
|---|---|
| `rl/obs_encoding.py` | `unit_slots` → `(feats[U,11], role_ids[U], valid, own_mask)`. `enemy_slots` + concat. |
| `rl/roles.py` | `ROLE_VOCAB: dict[str,int]` estable (sorted roles + `misc=último` o `0=pad`). Sembrar en `Vocab.seed_roles` no hace falta: vocab de **ítems** ≠ vocab de **entidades**. Lista propia, congelada. |
| `rl/network.py` | dims, `role_emb`, `adapt_capa2c_state_dict`, pool solo own, `dist_unit` enmascara enemigos. |
| `rl/rollout.py` | batch keys `unit_role_ids`, `unit_own_mask`. |
| `rl/action_adapter.py` | `unit_ids` **solo propias** (índice de cabeza 2 = slot own, no el concat). Mapear slot de red ↔ actor_id con `own_mask`. |
| `rl/trainer.py` | llamar `adapt_capa2c_state_dict` si `unit_mlp.0.weight` in_features discrepa. |

### Tests

- `e1` y `e3` → `role_id` distintos (`infantry_basic` vs `infantry_antiarmor`).
- Enemigo `2tnk` entra con `team=1`, `role=tank_medium`.
- Net2Net: `unit_mlp.0.weight[:, :10]` igual al ckpt A; extra ≈0.
- `dist_unit` nunca samplea slot enemigo (mask −1e9).
- Pool GRU: zero enemies ⇒ `unit_vec` idéntico a A en un forward con `role_emb=0` y `team=0` (regresión).
- Params siguen &lt; 8M.

### Criterio para pasar a C

wr20 no se cae a 0. Visor: scatter/unidad condicionada sigue moviendo `last_push` con el sujeto. Histograma `train` puede empezar a mezclar roles si el easy saca vehículos; **no** es requisito. Si H colapsa → restore A.

---

## PR-C — `attack` elige actor, no “el más cercano”

Depende de B (sin tokens enemigos la cabeza no tiene a quién puntuar).

### Acción

Cadena de Capa 2:

```
P(tipo) → P(unidad propia | tipo) → P(celda | tipo, unidad) → P(ítem | tipo)
```

Suma, **solo si tipo == attack**:

```
→ P(enemy_slot | tipo, unidad, celda)
```

`TYPES_USE_ENEMY = {"attack"}`. Log_prob F3: sumar esa cabeza **solo** cuando está activa (mismo patrón que unidad/celda/ítem).

Scorer: cat(token_enemigo, token_propio_elegido, hidden, type_emb) → logit. Máscara: enemigos visibles; opcional sesgo a distancia de la celda elegida (legal = visible; el sesgo espacial lo aprende).

F1 / adapter: `target_actor_id = enemy_ids[enemy_slot]`. Si el slot es pad o el actor murió → degradar a `attack_move` a la celda (ya existe).

**No** usar `_nearest_enemy_at_cell` cuando hay `enemy_slot` válido. Dejarlo como fallback.

`army_attack_move` / `attack_move` **no** usan esta cabeza: el engine AutoTarget sigue. C no es “micro de 256”; es un click de focus fire cada 2 s.

### Código

| Archivo | Cambio |
|---|---|
| `rl/network.py` | `enemy_scorer`, `dist_enemy`, `_heads_used` + `use_e`, `evaluate_actions` / `_seq` / `act`. |
| `rl/action_adapter.py` | `index_to_command` lee `enemy_slot`; mask legal. |
| `rl/rollout.py` | guardar `enemy_slot` en el action dict. |
| Tests F3 | `attack` suma 4 términos; `move` no. |

Init: scorer 0 / xavier. Tronco cargado. Adam fresco en `enemy_scorer` basta; fresco global también (consistente con 1010).

### Criterio de cierre 2c

20 iters sin NaN. Fracción `attack` con `target_actor_id ≠ nearest` &gt; 0 (si es 0, la cabeza no se usa — revisar mask). wr20 vs easy ≥ piso B. Recién ahí se puede hablar de Capa 3 `hard`.

---

## Net2Net — no se entrena de cero

| PR | Carga | Qué nace en 0 | Qué se conserva |
|---|---|---|---|
| A | 1:1 | nada | todo 970 / best easy |
| B | pad `Linear` in_features + keys nuevas | `role_emb`, cols extra mlp/scatter/scorer | GRU, U-Net, xf, type/item/value, `xf_scale` |
| C | 1:1 tronco B | `enemy_scorer` | todo B |

No resetear `unit_xf_scale` en B/C. No cambiar `HIDDEN_DIM`. No tocar vocab de ítems (roles de producción ya estables).

Si un ckpt pre-B se carga en red B sin `adapt_capa2c`, `load_state_dict` strict=False deja mlp en random parcial → **obligatorio** el adapt, igual que Capa 2.

---

## Qué no hacer

- 128 o 256 slots “por si acaso”.
- One-hot de `e1`/`1tnk` en el tensor: el vocab de actor rompe facción; usar `rl.roles`.
- Meter enemigos en el **pool** del GRU.
- `attack` a actor **antes** de tokens enemigos (C sin B).
- A+B+C en un resume.
- Declarar Capa 2c “lista” porque wr global pegó 30% (es era, 4 eps/iter).
- 2b (QSA/GDN) en el mismo corte.
- Volver a `SUPPORT_ASSAULT` para simular counters.

---

## Relación con el plan 12 / 13

El 12 no se modifica. Capa 2 “hecha” en el 13 = pointer + scatter + xf sobre **propias anónimas**. 2c es el resto de esa fila (“equipo, HP, rol” + set que incluye al rival).

Capa 3 sigue siendo el **cuándo** el matchup importa (composición del rival cambia). 2c es el **con qué** lo ve. Implementar 2c vs easy: el easy ya mezcla raids y vehículos; beginner no habría enseñado nada.

---

## Checklist de corte (operador)

1. Cortar train. Archivar con `rl/archive_run.py` (nombre de era: easy 50t 970-…).
2. Anotar best easy (iter + wr20 + `defense_loss`). Resume ese, no un beginner 951.
3. Un PR. Tests `rl/tests/test_proc_cells_best.py` + los nuevos del spec.
4. Boot: print del régimen (`MACRO 50`, `MAX_UNITS 96` / `role+enemy` / `attack-actor`).
5. 20 iters. Si H/NaN/wr20→0: restore, no el siguiente PR.

*Guardado: 2026-08-31 — rama `exp/rl-2026-08-28-grok`. PR-A shipped (96 + combat-first, resume 976). B y C pendientes.*
