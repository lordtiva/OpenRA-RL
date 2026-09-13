# Onboarding: de 0 a ~50% vs easy, sin un `best.pt`

> **Para quién:** clonaste el repo y **no tenés checkpoints**.
> **Qué no es:** un atajo para medir una arch nueva contra un run viejo. Eso sigue siendo `--scratch` vs 100% easy (sin `--onboard`).
> **Contrato Aliados:** [`../contract/ra-aliados.md`](../contract/ra-aliados.md). Docker / dash: [`operacion.md`](operacion.md).

Comando: `.\.venv\Scripts\python.exe rl\auto_train.py --scratch --onboard` (después del `--build` de Docker).

<details>
<summary>Notas de corte (mental-base / army push / Phase A) — no el hello-world</summary>

**2026-09-06 — mental base v3:** beacon GPS sigue apagado (`beacon=None` en encode/push). El teacher recuerda un **mental enemy-base** (centroide del cluster denser de edificios enemigos vistos) y empuja ahí cuando no hay leftover visible. Escalares: `has_enemy_base_belief`, rel dx/dy, conf (`SCALAR_DIM=33`, Net2Net pad). Phase A `a_max_steps` **1800**; early fog scout con ≥3 combate. Regenerar `teacher_wins/` (schema `eco_and_combat_scout_v1` / `--scratch --onboard`). **K=2 eco+push** same macro-tick.

**Army push (runtime adapter, 2026-09-07):** pipeline en `index_to_command_effective` + hysteresis en live/rollout. Live/eval lo toma al **reiniciar el proceso** (sin `--scratch`).

1. `stage_army_attack_cell`: always-on safety (like remap). Vanguard = units closer to dest than centroid (no `x>35` GPS). Flank N/S **solo** si el midline es agua.
2. `remap_move_cell`: `nearest_passable` cerca del click — **sin** funnel south-flank/ore. Fallback ilegal: contacto / `war_objective`, nunca beacon.
3. hysteresis (`should_emit_army_push` / `filter_army_push_hysteresis`, eps=8): no spamea el mismo `army_attack_move`.
4. `reject_feet_push_cell`: always-on. AM de grupo sobre el centroide/yard con pack idle ≥12 se reescribe a `war_objective` (raid/visible/mental/fog).
5. `guard_army_push_cell`: annealable (`heuristic_p`). No tira una vanguardia **lejos del objetivo** hacia el yard. Spawn-agnóstico.

**2026-09-13 — dest agnóstico / spawn aleatorio:** `war_objective` compartido (teacher + adapter + tape). Nunca `BEACON_BY_MAP` / `(95,11)`. Opening = fog scout desde el conyard. `--spawn random` (SW|NE por episodio), `--player-faction RandomAllies`, `--enemy-faction Random`. Sin map pool. Schema A/B `eco_and_combat_scout_v1`, C/D/E `eco_and_combat_expand_v3`. Requiere `--scratch --onboard` (cintas v4 no hidratan).

**2026-09-06 — Phase A inactivity:** Phase A / `--onboard` forces `--qsa-topk 0` and `--xf-topk 0` (dense). `balance_bc_samples` defaults **512/512**. Teacher `_push_cell` = `war_objective`. `bc_only` eval usa **temperature=0.0**.

</details>

---



## En una frase



El camino más corto que **viaja con el git** (sin pesos) es:

1. **A** — clonar al `ScriptedTeacher` **rush** vs `beginner` (SFT, sin PPO)
2. **B** — PPO + SIL + BC rifle vs `beginner` hasta wr20 ~50%
3. **C** — teacher **expand** (weap + `1tnk` + mix `e3`) + PPO vs `easy` (mix beginner→easy)
4. **D** — lo mismo vs `medium` (mix easy→medium)
5. **E** — lo mismo vs `hard` (OpenRA `normal`; mix medium→hard) hasta wr20 ~50%

Eso es lo que hace `auto_train.py --scratch --onboard`. Un solo comando. El experto es código (`rl/scripted_teacher.py`). `hard` en este repo es el bot **normal** de OpenRA, no un ModularBot llamado hard.



---



## Requisitos



Desde `C:\Users\lordc\Desktop\OpenRA-RL` (o el path del clone):



```powershell

cd C:\Users\lordc\Desktop\OpenRA-RL

$env:PYTHONPATH=""

docker compose up -d --build openra-rl

curl.exe -f http://localhost:8000/health

```



Esperá `200`. El segundo daemon (`docker-compose.scale.yaml`, puerto 8010) es opcional; `auto_train` usa los que respondan.



Detalle de Docker / dashboard / cuelgues: [`operacion.md`](operacion.md).



---



## El comando



**Primera vez (pesos aleatorios, curriculum nuevo):**



```powershell

cd C:\Users\lordc\Desktop\OpenRA-RL

$env:PYTHONPATH=""

.\.venv\Scripts\python.exe rl\auto_train.py --scratch --onboard

```



**Si se cortó (Ctrl+C, crash, corte de luz):**



```powershell

.\.venv\Scripts\python.exe rl\auto_train.py --onboard

```



Retoma la fase que quedó en `rl/ckpts/curriculum.json`. **No** pases `--scratch` o reinicia en A.



Ctrl+C para parar. El log: `rl/auto_train.log`.



### Flags opcionales (defaults entre paréntesis)



| Flag | Default | Qué cambia |

|---|---|---|

| `--onboard-sft-iters` | 20 | Iters de SFT en A |

| `--onboard-a-promote-wr20` | 0.25 | wr20 vs beginner para pasar A→B (**sin** streak) |

| `--onboard-promote-wr20` | 0.50 | wr20 vs beginner para pasar B→C (× streak) |

| `--onboard-done-wr20` | 0.50 | wr20 vs hard para marcar DONE |

| `--onboard-streak` | 10 | Iters **seguidos** sobre el umbral (B y C; A no usa streak) |

| `--onboard-min-iters` | 20 | Mínimo de iters en B y en C (un 4/4 suelto no promociona) |

| --onboard-rewind N | — | **Una vez**: latest/est ← iterN, trunca metrics/race, luego **recomputa promote** con el jsonl quedado (A+wr20→B; B streak+wr20→C; etc). N≤sft_iters parte en A (puede promover). C→B undo puede re-promover si el streak B sigue claro. |
| `--onboard-rush` | 8 | `ScriptedTeacher.RUSH_ATTACK_MOVE`. Bench n=20 eligió 8; 6/5 suben lose_rate. |
| `--onboard-bc-games` | 4 | Partidas teacher / iter en A. |
| `--onboard-eval-games` | 4 | Partidas eval del alumno / iter en A. |
| `--onboard-fresh-tapes` | — | Borra `teacher_wins/` y re-juega al teacher. |
| `--onboard-collect` | — | Suma teacher games aunque ya haya tapes. |
| `--onboard-collect-only` | — | Solo acumula `teacher_wins` hasta `--onboard-collect-target` (default 40); sin SFT/eval; no borra `latest.pt`; no requiere `--scratch`. |

`--scratch --onboard`, el **resume** `--onboard` y el promote **A→B** reusan `teacher_wins/` si el schema coincide: A/B `eco_and_combat_scout_v1`, C/D/E `eco_and_combat_expand_v3`. Tope 40 eps (las más cortas pisan las más largas). Al promover B→C (y C→D, D→E) se **borran** las cintas: el rifle teacher no se clona vs easy. `--onboard-collect` / `--onboard-collect-only` siguen re-jugando. Cintas con schema distinto se ignoran (v4/expand_v2 no hidratan).



---



## Qué va a pasar (y qué vas a leer en el log)



Estado en `rl/ckpts/curriculum.json`. Cada salto mata el `rl.train` y lo relanza con otros flags; no cambia de rival a mitad de un proceso.



### Fase A — SFT + eval del alumno



- Rival del **teacher**: `beginner`. El scripted **no** clona el ModularBot

  hard (`IOrder` ≠ `ActionIndex`). Copia el *timing* de rush (atacar en cuanto

  hay squad; beginner tiene `SquadSize: 3` y `MinimumAttackForceDelay: 150000`)

  y la eco de beginner (1–2 harvs, 1 barracks). Hard/normal (4 harvs + weap +

  squad 20) no entra en `a_short` @53k.

- Teacher: proc antes de tent, **sin weap** en el camino crítico. A 8 rifles

  `attack_move` de todo el idle (legal sin pack). A 12, `army_attack_move`.

  Hunt map-agnostic (`war_objective`): home raid → leftover visible → **mental
  enemy-base** (cluster denser de edificios vistos) → `last_seen` / belief
  ghosts → hunt/sweep cerca del ultimo contacto → fog scout. **Nunca**
  `resolve_beacon` / `BEACON_BY_MAP`. Encode Ch7-8 GPS off (`beacon=None`).
  Early fog scout away from home con >=3 combate antes del rush.
  Blob piled lejos de casa sin contacto → remate hunt.
  Train: `--spawn random` (SW|NE), `--player-faction RandomAllies`,
  `--enemy-faction Random`. Sin map pool.

  Phase A also: dense QSA/XF (`--qsa-topk 0 --xf-topk 0`), BC caps 512/512,
  greedy student eval (`temperature=0.0`). Regenera `teacher_wins/` (schema
  `eco_and_combat_scout_v1` / `--scratch --onboard`; cintas v4 no
  se auto-cargan).

- Clona cintas **`win` only** (`--bc-only`). **No** clona `lose` ni `incomplete` (timeout turtle).

  Opening: TRAIN/BUILD. Attack (leftover o ≥8 combate): **también** un push

  (`army_attack_move` / `attack_move`). **K=2 eco+push** same macro-tick (BC labels + student dual-emit). Sin eco no hay army; sin push el SFT es miller.

- **TeacherWinBuffer** persistente: acumula wins entre iters bajo `{ckpt_dir}/teacher_wins/`

  (`manifest.json` + `ep_XXXX.pt`). Tope **40 episodios** (`--bc-win-ep-cap`); una win más corta pisa la más larga. `--bc-win-cap 64000` es backstop de steps. **A las 20 wins** (`--bc-replay-at 20`) las iters siguientes **reusan el ring** (no abren partidas teacher). Promote A→B también pasa `--bc-replay`. `--onboard-collect` / `--onboard-collect-only` siguen llenando (`--bc-replay-at 0`). Wins ≥`--bc-win-prefer-ticks` 20000 se recortan primero. Override con `--bc-win-dir`.

- Además, 4 partidas del **alumno** por iter (`--eval-games 4`, sin PPO) para

  medir wr20 del clone. Sin eso A promocionaba a las 20 iters con un miller.

- Macro **40** ticks / **1800** decisiones. **4** teacher + **4** eval. 6 epochs NLL. BC **wins-only** (no clona incompletes; rush teacher=8).

- Collapse **off**. Hang 1200 s.



En el log: `onboard START phase A`, `BC-ONLY`, `[bc] keep wins=…`, `[eval] student`.



Criterio de salida: ≥ `--onboard-sft-iters` **y** wr20 del alumno ≥

`--onboard-a-promote-wr20` (default **0.25**) sobre las últimas 20 partidas.

**Sin** streak de 10 (eso queda solo para B→C). Log: `onboard PROMOTE A -> B`.



Si A ya corrió 20 iters de SFT viejo: --onboard --onboard-rewind 20

restaura iter0020.pt y trunca metrics. Si el wr20 del alumno en esas

filas ya ≥ --onboard-a-promote-wr20, el curriculum **queda en B**

(no hace falta rellenar wr20 de cero). Sin outcomes / wr bajo, sigue en A.

Resume --onboard sin rewind también aplica promotes ya ganados en metrics.



### Fase B — PPO + teacher BC vs beginner



- Resume `latest.pt` de A.

- `--bot-type beginner --sil` **y** `--bc --bc-teacher-bot beginner`.

  2 partidas teacher / iter (APM de A: macro 40 / 1000), 2 epochs NLL,

  `λ_bc` 1→**0.10** en 40 iters (no se apaga). BC **wins-only** (sin

  `--bc-keep-incomplete`). Sin `--bc-only` (PPO sigue, salvo tanda wipe:

  4 lose <15k ticks; ahí solo BC/SIL). Sin PFSP.

- `--no-amp` (igual C/D/E). `auto_train --no-amp` también lo acepta y lo reenvía a `rl.train` (útil para forzar en A).

- PPO: macro 50 / max-steps 1000, `--lr 2.0e-5`, `--adv-mode global`.

  El teacher de B **no** usa esos knobs.

- Collapse **on** en B (mismo `--collapse` default que C). Restaura `best.pt`

  si política muerta o sequía wr20. `--no-collapse` lo apaga. Fase A lo

  ignora siempre (un eval suerte congelaría el SFT).

- Remate tardío (`SUPPORT_LATE_REMNANT`): tick≥25k y niebla vacía, sweep a

  bordes (no beacon). Sigue con `--no-war-nudge`.



Métrica norte: **wr20 vs beginner**, no el `iter_winrate` de una tanda de 4.



Criterio: wr20 ≥ 0.50 durante 10 iters seguidos **y** al menos 20 iters en B.



Log: `onboard PROMOTE B -> C`. En metrics aparece una línea `era_reset` para que el wr vs beginner **no** se hidrate como wr vs easy.



### Rewind de un B en wipe (volver a `best@24`)



Si B ya escribió un wipe encima de A (`latest` ≫ `best`, wr20=0, harvest

edge muy negativo): **no** relances `--onboard` sobre ese `latest`. Ctrl+C

y, **una vez**:



```powershell

.\.venv\Scripts\python.exe rl\auto_train.py --onboard --onboard-rewind 24

```



Eso:



| Archivo | Qué hace |

|---|---|

| `best.pt` / `best.json` | También ← `iterN` (si no, collapse restauraría un C 0-win) |

| `latest.pt` | Copia de `best.pt` si `best.json.iter==N`, si no `iterN.pt` |

| `metrics.jsonl` | Tira filas con `iter>24` (conserva A y `era_reset`) |

| `economy_race.jsonl` | Igual, `iter>24` afuera (si no, el dash sigue gritando harvest −600) |

| `curriculum.json` | Fase B (desde C también vuelve a B). `b_bc_start_iter` queda en el origen de B (`λ_bc` en el piso). Desde C, `phase_started_iter=N` para que `min_iters` no re-promueva al toque. |



No hace falta `--scratch`. Collapse en B queda ON; `--no-collapse` si no

querés restore de `best.pt`.

Siguiente corte de luz: `--onboard` **sin** rewind.



No borres `iter0030.pt`… a mano: el train los pisa al subir. `live_games.jsonl`

/ `live_tape.jsonl` / `auto_train.log` no entran al wr; se pueden dejar.



Si el seed que querés es `iter0140.pt` y no el `best@24`: `--onboard-rewind 140`. Desde C, el mismo flag vuelve a B (p.ej. `--onboard-rewind 130`).



### Fase C — PPO + expand BC vs easy

Al promover B→C se copia `best.pt` → `best_B.pt`, se **wipea** `teacher_wins/` (schema expand) y λ_bc vuelve a warmup desde el start de C.

- Teacher **expand**: mismo opening que A/B, después del blob (`n_combat≥8`) FORCE `weap` → `1tnk` (luego `2tnk`), mix `e3`, un pbox. No mill 3ª proc. No APC.
- `--bc --bc-teacher-bot easy --bc-teacher-mode expand`. Wins-only. El rifle teacher **no** entra acá.
- Mix `--mix-from beginner` 0.25→1.0 en 40 iters. wr20 / best / promote **solo vs easy**.
- `--no-amp`. Sequía bloqueada hasta `min_iters`.

Criterio: wr20 vs **easy** ≥ 0.45 × 10 iters, mínimo 20 iters → D.

### Fase D — vs medium

Igual que C con expand BC vs `medium`, mix easy→medium. Snapshot `best_C.pt`. Criterio wr20 ≥ 0.45 × 10, min 20 → E.

### Fase E — vs hard (DONE)

`--bot-type hard` (OpenRA **normal**). Mix medium→hard. Expand BC vs hard: si el teacher gana poco, el buffer se llena lento y manda SIL+mix (igual que el C viejo sin maestro). Criterio: wr20 vs **hard** ≥ 0.50 × 10, min 20.

Log: `onboard PROMOTE E -> done`. `auto_train` sale. El agente útil es `rl/ckpts_v2/best.pt`.



---



## Cuánto tarda (orden de magnitud, 2070 + 5600X)



No es un número de paper. Es wall-clock de sótano:



| Fase | Qué limita | Orden |

|---|---|---|

| A | 20 iters × teacher + eval | unas horas |
| B | wr20 vs beginner a 50% | de un día a varios |
| C | expand + wr20 vs easy 45% | otro tanto; easy tiene weap/cajas |
| D | vs medium | más; medium rushea a los 5 s |
| E | vs hard (normal) a 50% | el tramo largo; no es un 50% de beginner |



Si B no llega a 50% en ~200 iters, el teacher de A no dejó un build order usable, o el entorno (Docker / `a_short` / auto-support) no está como el de este repo. No subas a easy a mano.



---



## Qué **no** hacer



- `--onboard` con un `latest.pt` de otro experimento y **sin** `curriculum.json`. El launcher se niega: archivá primero (`rl/archive_run.py`) y usá `--scratch --onboard`.

- `--scratch --onboard` a mitad de B–E. Reinicia en A y tira el curriculum.

- Meter `--pfsp` / rl en este camino. El overlay de fase **saca** PFSP de `TRAIN_ARGS`. Hard es la fase E, no un flag a mano.

- Clonar el teacher **rush** vs easy/hard. A/B son rifle; C/D/E son expand. El launcher wipea tapes al promover.

- Relanzar `--onboard` sobre un `latest` de wipe sin `--onboard-rewind`. `λ_bc` ya sería 0 y los pesos son el atractor mill.

- Editar `TRAIN_ARGS` para “hacer el onboard a mano” mezclando A y C en el mismo proceso. Hidrata wr de beginner en easy y miente el dash.

- Comparar este reloj con “Run 46 en 20 iters”. Run 46 arrancó de un best de Run 45.



---



## Cómo se relaciona con el resto del proyecto



| Pregunta | Comando |

|---|---|

| Clone vacío → jugar a easy | `--scratch --onboard` (este doc) |

| ¿Cuánto tarda *esta* arch desde 0 vs easy, sin teacher? | `--scratch` **sin** `--onboard`, `--bot-type easy`, sin PFSP |

| Seguir un run maduro (PFSP, QSA, etc.) | `auto_train.py` sin `--onboard` (usa `TRAIN_ARGS`) |

| ¿SFT del bot hard de OpenRA? | No está. Hay que instrumentar C# (`IOrder` → `ActionIndex`). El atajo publicable es el teacher Python. |



Detalle de operación diaria: [`operacion.md`](operacion.md). Plan de capas (BC/SIL/self-play): [`../design/plan-4-capas.md`](../design/plan-4-capas.md).

Gaps RA completo (naval/air/buildings/micro): [`../contract/ra-completo.md`](../contract/ra-completo.md). Contrato Aliados: [`../contract/ra-aliados.md`](../contract/ra-aliados.md).



---



## Archivos que toca el onboard



| Path | Rol |

|---|---|

| `rl/ckpts/curriculum.json` | Fase actual, umbrales, `b_bc_start_iter` |

| `rl/ckpts/metrics.jsonl` | wr20; líneas `era_reset` al cambiar de fase |

| `rl/ckpts/economy_race.jsonl` | Harvest del dash; se trunca con `--onboard-rewind` |

| `rl/ckpts/latest.pt` / `best.pt` | pesos |
| `rl/ckpts/teacher_wins/` | ring BC persistente (wins del ScriptedTeacher) |

| `rl/onboard.py` | Scheduler puro (testeable, sin Docker) |

| `rl/auto_train.py --onboard` | Mata / relanza `rl.train` por fase |

| `rl/train.py --bc-only` `--bc-teacher-bot` `--bc-games` `--bc-epochs` | SFT (A) y BC mixto (B) |



Tests: `python rl/tests/test_onboard.py` y `python rl/tests/test_imitation.py`.

