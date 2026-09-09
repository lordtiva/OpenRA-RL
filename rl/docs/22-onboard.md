# 22 — Onboarding: de 0 a ~50% vs easy, sin un `best.pt`



> **Para quién:** alguien que clonó el repo y **no tiene checkpoints**. No hace falta un `best.pt` de un run anterior.

> **Qué no es:** un atajo para medir una arquitectura nueva contra Run 46. Eso sigue siendo `--scratch` vs 100% easy (ver más abajo).

> **Fecha:** 2026-09-04.


> **2026-09-06 — mental base v3:** beacon GPS sigue apagado (`beacon=None` en
> encode/push). El teacher recuerda un **mental enemy-base** (centroide del
> cluster denser de edificios enemigos vistos) y empuja ahí cuando no hay
> leftover visible. Escalares nuevos: `has_enemy_base_belief`, rel dx/dy, conf
> (`SCALAR_DIM=29`, Net2Net pad). Phase A `a_max_steps` **1800**; early fog
> scout con ≥3 combate. Regenerar `teacher_wins/` (schema
> `eco_and_combat_mental_v4` / `--onboard-fresh-tapes`). **K=2 eco+push** same macro-tick (student dual-emit + BC labels).
- **Army push (runtime adapter, 2026-09-07):** pipeline en `index_to_command_effective` + hysteresis en live/rollout. Live/eval lo toma al **reiniciar el proceso** (sin `--scratch`). Un live WIN (~40k ticks a base NE) valida el remate; Phase A promotion sigue necesitando wr20 en metrics.
  1. `stage_army_attack_cell`: `n_advanced>=8` / `cen.x>35` / attractor guard — flank N/S **solo** en opening choke.
  2. `remap_move_cell`: `nearest_passable` cerca del click — **sin** funnel south-flank/ore.
  3. hysteresis (`should_emit_army_push` / `filter_army_push_hysteresis`, eps=8): no spamea el mismo `army_attack_move`.
  4. `guard_army_push_cell`: no tira al oeste una vanguardia `x>70`; fog-east retarget (beacon/mental/front) si no hay edificios enemigos visibles.



> **2026-09-06 — Phase A inactivity fixes:** Phase A / `--onboard` forces
> `--qsa-topk 0` and `--xf-topk 0` (dense; sparse topk was masking enemy push
> cells and dropping attack BC). `balance_bc_samples` defaults **512/512**
> (was ~96). Teacher `_push_cell`: visible/leftover/ghost/mental-base first;
> only when belief empty, `resolve_beacon` is the opening-SFT prior, else fog
> scout. `bc_only` eval uses **temperature=0.0**. Schema
> `eco_and_combat_mental_v4` — regenerate tapes (`--onboard-fresh-tapes`).
> Keep `latest.pt` (no arch/SCALAR change).




---



## En una frase



El camino más corto que **viaja con el git** (sin pesos) es:



1. **A** — clonar al `ScriptedTeacher` vs `beginner` (SFT, sin PPO)

2. **B** — PPO + SIL + BC del teacher vs `beginner` hasta que wr20 se sostenga ~50%

3. **C** — los mismos pesos vs `easy` hasta wr20 ~45% (sin teacher)



Eso es lo que hace `auto_train.py --scratch --onboard`. No clona un best viejo. El experto es código (`rl/scripted_teacher.py`).



Run 46 (~45% wr20 vs easy) **no** nació así: venía de una cadena de resumes. El onboarding mide *otro* reloj: horas desde un clone vacío hasta un agente que le gana a easy la mitad de las veces.



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



Detalle de Docker / dashboard / cuelgues: `07-operacion.md`.



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

| `--onboard-done-wr20` | 0.45 | wr20 vs easy para marcar DONE |

| `--onboard-streak` | 10 | Iters **seguidos** sobre el umbral (B y C; A no usa streak) |

| `--onboard-min-iters` | 20 | Mínimo de iters en B y en C (un 4/4 suelto no promociona) |

| `--onboard-rewind N` | — | **Una vez**: `latest` y `best` ← `iterN`/`best@N`, trunca metrics/race. N≤`sft_iters` vuelve a A. N>sft en **B o C** vuelve a B; `λ_bc` queda en el piso. |
| `--onboard-rush` | 8 | `ScriptedTeacher.RUSH_ATTACK_MOVE`. Bench n=20 eligió 8; 6/5 suben lose_rate. |
| `--onboard-bc-games` | 4 | Partidas teacher / iter en A. |
| `--onboard-eval-games` | 4 | Partidas eval del alumno / iter en A. |
| `--onboard-fresh-tapes` | — | Borra `teacher_wins/` y re-juega al teacher. |
| `--onboard-collect` | — | Suma teacher games aunque ya haya tapes. |
| `--onboard-collect-only` | — | Solo acumula `teacher_wins` hasta `--onboard-collect-target` (default 40); sin SFT/eval; no borra `latest.pt`; no requiere `--scratch`. |

`--scratch --onboard` y el **resume** `--onboard` (sin `--scratch`) **reusan** `teacher_wins/` si el schema coincide (`eco_and_combat_mental_v4`) y hay eps; `launch_train` pasa `--bc-replay` (fases **A y B**; no C). Log: `reusando teacher_wins/ … resume no re-juega al teacher` y en train `[bc] replay tapes … (no collect)`. `--onboard-collect` / `--onboard-collect-only` siguen re-jugando. Cintas viejas (`eco_and_combat_v1` / hunt_v2 / sin schema) se ignoran y se vuelven a recolectar **una vez**.



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

  Hunt map-agnostic: home raid → leftover visible (micro) → **mental
  enemy-base** (cluster denser de edificios vistos) → `last_seen` / belief
  ghosts → hunt/sweep cerca del ultimo contacto → **`resolve_beacon` as
  opening-SFT prior when belief empty** → fog scout. Encode Ch7-8 GPS still
  off (`beacon=None`); beacon only labels teacher push when there is no
  belief. Early fog scout away from home con >=3 combate antes del rush.
  Blob piled lejos de casa sin contacto → remate hunt.

  Phase A also: dense QSA/XF (`--qsa-topk 0 --xf-topk 0`), BC caps 512/512,
  greedy student eval (`temperature=0.0`). Regenera `teacher_wins/` (schema
  `eco_and_combat_mental_v4` / `--onboard-fresh-tapes`; cintas v3/hunt_v2 no
  se auto-cargan).

- Clona cintas **`win` only** (`--bc-only`). **No** clona `lose` ni `incomplete` (timeout turtle).

  Opening: TRAIN/BUILD. Attack (leftover o ≥8 combate): **también** un push

  (`army_attack_move` / `attack_move`). **K=2 eco+push** same macro-tick (BC labels + student dual-emit). Sin eco no hay army; sin push el SFT es miller.

- **TeacherWinBuffer** persistente: acumula wins entre iters bajo `{ckpt_dir}/teacher_wins/`

  (`manifest.json` + `ep_XXXX.pt`). Cada update BC samplea el ring completo (un iter

  con 0 wins nuevos sigue entrenando). Cap default `--bc-win-cap 64000`

  (~40-60 short rushes before trim; eco+push K=2 ~1.0-1.5k steps). Wins ≥`--bc-win-prefer-ticks` 20000 se recortan

  primero (un 33k miller-win no come el dataset). SIL sigue en 40k. Override

  con `--bc-win-dir`.

- Además, 4 partidas del **alumno** por iter (`--eval-games 4`, sin PPO) para

  medir wr20 del clone. Sin eso A promocionaba a las 20 iters con un miller.

- Macro **40** ticks / **1800** decisiones. **4** teacher + **4** eval. 6 epochs NLL. BC **wins-only** (no clona incompletes; rush teacher=8).

- Collapse **off**. Hang 1200 s.



En el log: `onboard START phase A`, `BC-ONLY`, `[bc] keep wins=…`, `[eval] student`.



Criterio de salida: ≥ `--onboard-sft-iters` **y** wr20 del alumno ≥

`--onboard-a-promote-wr20` (default **0.25**) sobre las últimas 20 partidas.

**Sin** streak de 10 (eso queda solo para B→C). Log: `onboard PROMOTE A -> B`.



Si A ya corrió 20 iters de SFT viejo: `--onboard --onboard-rewind 20` vuelve

a fase A desde `iter0020.pt` (no uses 24).



### Fase B — PPO + teacher BC vs beginner



- Resume `latest.pt` de A.

- `--bot-type beginner --sil` **y** `--bc --bc-teacher-bot beginner`.

  2 partidas teacher / iter (APM de A: macro 40 / 1000), 2 epochs NLL,

  `λ_bc` 1→**0.10** en 40 iters (no se apaga). BC **wins-only** (sin

  `--bc-keep-incomplete`). Sin `--bc-only` (PPO sigue, salvo tanda wipe:

  4 lose <15k ticks; ahí solo BC/SIL). Sin PFSP.

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



### Fase C — PPO vs easy (rampa, sin BC)



Al promover B→C se copia `best.pt` → `best_B.pt`. C **no** pisa ese best con un 0-win vs easy.



- Mismos pesos, **mismos knobs que B**: `--lr 2.0e-5`, `--adv-mode global`, Adam de B (**sin** `--reset-opt`).
- **Sin BC.** El teacher no es experto vs easy; clonarías perder. El ancla es SIL (elite persistido en `elite.pt`) + mix de rival.
- **Rampa suave** `--mix-from beginner`: P(easy) sube de **0.25 → 1.0** en 40 iters. wr20 / best.pt / promote **solo vs easy**. Beginner wins llenan SIL al arrancar.
- **C must not use AMP** (--no-amp): skip_frac=1.0 killed learning. Do not pass --amp-init-scale on C.
- Sequía: ignora el pico wr20 de B. En C no restaura hasta `min_iters` **y** un pico propio vs easy. H=0 de PPO skipped no cuenta como política muerta si el rollout sigue entrenando/raseando.

El wr vs easy se cae al principio. Es normal (Run 11: salto a easy = 0/140). Lo que **no** es normal es H=clip=gn=0 durante decenas de iters — eso es PPO skipeado, no transferencia.

Criterio: wr20 vs **easy** ≥ 0.45 × 10 iters, mínimo 20 iters en C.



Log: `ONBOARD DONE`. `auto_train` sale. El agente útil es `rl/ckpts/best.pt` (y `latest.pt`).



---



## Cuánto tarda (orden de magnitud, 2070 + 5600X)



No es un número de paper. Es wall-clock de sótano:



| Fase | Qué limita | Orden |

|---|---|---|

| A | 20 iters × 2 partidas teacher | unas horas |

| B | wr20 vs beginner a 50% | de un día a varios (el techo histórico vs beginner es alto) |

| C | recuperar wr vs easy | otro tanto; easy pega en casa |



Si B no llega a 50% en ~200 iters, el teacher de A no dejó un build order usable, o el entorno (Docker / `a_short` / auto-support) no está como el de este repo. No subas a easy a mano.



---



## Qué **no** hacer



- `--onboard` con un `latest.pt` de otro experimento y **sin** `curriculum.json`. El launcher se niega: archivá primero (`rl/archive_run.py`) y usá `--scratch --onboard`.

- `--scratch --onboard` a mitad de B/C. Reinicia en A y tira el curriculum.

- Meter `--pfsp` / hard / rl en este camino. El wr20 dejaría de ser el reloj de onboarding. El overlay de fase **saca** PFSP de `TRAIN_ARGS`.

- Activar `--bc` vs easy (fase C). El scripted no es experto vs easy; clonarías perder. A y B clonan vs **beginner**.

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



Detalle de operación diaria: `07-operacion.md`. Plan de capas (BC/SIL/self-play): `12-plan-4-capas-siguiente-nivel.md`.

Gaps RA completo (naval/air/buildings/micro): `23-ra-completo-todo.md`.



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

