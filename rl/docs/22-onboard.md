# 22 — Onboarding: de 0 a ~50% vs easy, sin un `best.pt`

> **Para quién:** alguien que clonó el repo y **no tiene checkpoints**. No hace falta un `best.pt` de un run anterior.
> **Qué no es:** un atajo para medir una arquitectura nueva contra Run 46. Eso sigue siendo `--scratch` vs 100% easy (ver más abajo).
> **Fecha:** 2026-09-04.

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
| `--onboard-promote-wr20` | 0.50 | wr20 vs beginner para pasar a C |
| `--onboard-done-wr20` | 0.45 | wr20 vs easy para marcar DONE |
| `--onboard-streak` | 10 | Iters **seguidos** sobre el umbral |
| `--onboard-min-iters` | 20 | Mínimo de iters en B y en C (un 4/4 suelto no promociona) |
| `--onboard-rewind N` | — | **Una vez**: `latest` ← `best`/`iterN`, trunca metrics/race. N≤`sft_iters` vuelve a A (`iter0020.pt`). N>sft en B pinnea BC. |

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
  Leftover visible > beacon; peel de raid; TRAIN e1 en el push.
- Clona cintas `win` y `incomplete` largos (`--bc-only`). **No** clona `lose`.
- Además, 4 partidas del **alumno** por iter (`--eval-games 4`, sin PPO) para
  medir wr20 del clone. Sin eso A promocionaba a las 20 iters con un miller.
- Macro 20 ticks / 2800 decisiones. 4 teacher + 4 eval en el pool. 6 epochs NLL.
- Collapse **off**. Hang 1200 s.

En el log: `onboard START phase A`, `BC-ONLY`, `[bc] keep wins=…`, `[eval] student`.

Criterio de salida: ≥ `--onboard-sft-iters` **y** wr20 del alumno ≥ 0.50 × 10
iters (mismo umbral que B). Log: `onboard PROMOTE A -> B`.

Si A ya corrió 20 iters de SFT viejo: `--onboard --onboard-rewind 20` vuelve
a fase A desde `iter0020.pt` (no uses 24).

### Fase B — PPO + teacher BC vs beginner

- Resume `latest.pt` de A.
- `--bot-type beginner --sil` **y** `--bc --bc-teacher-bot beginner`.
  4 partidas teacher / iter (APM de A: macro 20 / 2800), 2 epochs NLL,
  `λ_bc` 1→**0.25** en 80 iters (no se apaga). Clona `win` **e** `incomplete`
  largo — el opening de A eran esas cintas, no solo wins. Sin `--bc-only`
  (PPO sigue, salvo tanda wipe: 4 lose <15k ticks; ahí solo BC/SIL).
  Sin PFSP.
- PPO: macro 50 / max-steps 1000. El teacher de B **no** usa esos knobs.
- Collapse **off** en B. Un `best.pt` con iwr 0.5 (un 2/4 suelto) congela el
  puntero; 3 iters `deploy-noop` restauraban ese snapshot cada ~20 iters y
  nunca salías del SFT. Si la política se muere, se sigue; el criterio de
  salida es wr20, no el watchdog.

B **sin** clonar incompletes del teacher se olvida del opening (mill / wipe,
wr20=0, clip alto, `bc_n=0` casi siempre). El mix es el default.

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
| `best.pt` / `best.json` | No los toca (son el seed) |
| `latest.pt` | Copia de `best.pt` si `best.json.iter==24`, si no `iter0024.pt` |
| `metrics.jsonl` | Tira filas con `iter>24` (conserva A y `era_reset`) |
| `economy_race.jsonl` | Igual, `iter>24` afuera (si no, el dash sigue gritando harvest −600) |
| `curriculum.json` | Sigue en B; pinnea `b_bc_start_iter=24` para que `λ_bc` arranque en 1.0 |

No hace falta `--scratch` ni `--no-collapse` (A/B ya apagan collapse).
Siguiente corte de luz: `--onboard` **sin** rewind.

No borres `iter0030.pt`… a mano: el train los pisa al subir. `live_games.jsonl`
/ `live_tape.jsonl` / `auto_train.log` no entran al wr; se pueden dejar.

Si el seed que querés es `iter0140.pt` y no el `best@24`: `--onboard-rewind 140`.

### Fase C — PPO vs easy

- Mismos pesos, `--bot-type easy`, Adam fresco **solo** en el primer launch de C.
- El wr se va a caer. Es normal (Run 11: salto prematuro a easy = 0/140). Acá ya venís de un beginner estable.

Criterio: wr20 ≥ 0.45 × 10 iters, mínimo 20 iters en C.

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

---

## Archivos que toca el onboard

| Path | Rol |
|---|---|
| `rl/ckpts/curriculum.json` | Fase actual, umbrales, `b_bc_start_iter` |
| `rl/ckpts/metrics.jsonl` | wr20; líneas `era_reset` al cambiar de fase |
| `rl/ckpts/economy_race.jsonl` | Harvest del dash; se trunca con `--onboard-rewind` |
| `rl/ckpts/latest.pt` / `best.pt` | pesos |
| `rl/onboard.py` | Scheduler puro (testeable, sin Docker) |
| `rl/auto_train.py --onboard` | Mata / relanza `rl.train` por fase |
| `rl/train.py --bc-only` `--bc-teacher-bot` `--bc-games` `--bc-epochs` | SFT (A) y BC mixto (B) |

Tests: `python rl/tests/test_onboard.py` y `python rl/tests/test_imitation.py`.
