# Macro-first curriculum (2026-09)

Estado **implementado** en `feat/macro-first-curriculum` (ckpts `rl/ckpts_v2`).
Sustituye la doctrina rush→expand (desaprender rifles) por **expand desde el tick 0**,
reward/tech alineados, y arch práctica sin GTrXL.

Onboard operativo: [`../start/onboard.md`](../start/onboard.md).
Operación / VRAM: [`../start/operacion.md`](../start/operacion.md).

---



## eradicate_v6 (fixed Beginner→Hard+ MDP)

Default shaper as of 2026-09-18. One reward + one action meaning for all rivals
(PFSP only picks the bot — no per-difficulty harvester caps or weap cash-saves).

- **Timeout wipe:** on incomplete/truncated, terminal = -max(0, episode_return) - 2
  so turtle+mining cannot outscore a failed push.
- **No mining_rate farm:** w_mining_rate=0; keep small first_ore/refinery bootstrap.
- **Army-ratio delta:** dense Δ own/(own+ene) via orce_estimate.aoa_features 
el_power
  (0.5 under fog with no visible enemy — no free points).
- **Attack macro = war front:** *_attack_move resolves to war_objective unless a
  visible home raid exists (adapter, not a scripted war_nudge).
- **Crutches off:** should_save_for_weap always false; no train-harvester hard cap
  (extra harvs come from weap). **Anti-rush mask:** `BUILD weap` and 2nd `proc`
  stay illegal until tent/barr stands **and** `n_combat >= 4`. The net must
  sample barracks then rifles.
- **auto_support APM only:** repair, power_down, harvest-idle (no cell). No
  BUILD/PLACE/TRAIN/deploy/nudge/scout.

Preset: --shaper-preset eradicate_v6 (train/onboard default).



## Neuro-inspired A+B (ring + Kenyon novelty)

Optional defaults **on** (--ring-goal, --sil-novelty; disable with --no-*).

### B — Kenyon k-WTA novelty (
l/state_hash.py)
EliteBuffer / TeacherWinBuffer sample with novelty fingerprints instead of pure
even-pick. Metrics in metrics.jsonl:
- sil_diversity / elite_diversity: 
_unique_hash, unique_ratio, mean_pairwise_hamming

Contrast: same run with --no-sil-novelty and compare those fields + drought rate.

### A — RingGoalBias (
etwork.RingGoalBias)
Differentiable cell-logit bias from mental-base scalars (25–27) + GRU hidden.
Inside _logits_cell (in π), not a Python rewrite. Metrics:
- 
ing: 
ing_bias_mean, 
ing_bias_peak, 
ing_has_goal, 
ing_scale
- spatial_corr: 
edirect_rate of attack→war_objective (should fall as ring learns)

Contrast: --no-ring-goal and watch spatial_corr.redirect_rate + wr.

## Idea en una frase

El alumno **nunca** aprende spam `e1` como política base. El teacher es siempre
`mode="expand"`, el reward castiga oleadas suicidas y premia tier (`weap`/`dome`),
y la red ve tech/blindaje/amenaza + buildings en el XF.

---

## Currícula (3 etapas, labels A…E)

Los nombres A/B/S/C/D/E se conservan en `curriculum.json` / heartbeat.
Semántica continua:

| Etapa | Fases | Qué hace |
|-------|-------|----------|
| 1 Bootstrap macro | **A → B** | Teacher **expand** desde A. A = `--bc-only`. B = PPO+SIL+BC con λ_bc → **0.05**. Sin wipe de tapes al promover. |
| 2 Escalera suave | **S → C → D** | MAB / PFSP sobre bots scripted. Promote a C **sin** wipe elite/tapes; primer launch C **`--reset-opt --hyper-pause`** (Adam no hereda el piso 1e-5 de B/S). |
| 3 Liga | **E** | PFSP-RL + hist checkpoints + anclas hard/medium. |

`teacher_mode_for_phase(*)` → siempre `"expand"`.
Schema de tapes: `eco_and_combat_expand_v3` (acumulativo).
`wipe_elite` queda para **rewind/collapse**, no para promote sano.

---

## Agente / obs / reward / PPO

### Hecho

| Pieza | Dónde | Notas |
|-------|-------|-------|
| `eradicate_v5` default | `reward_shaping.py`, `train.py`, `TRAIN_ARGS` | Combate **lineal simétrico** (`w_kills=w_deaths=0.15`); `w_exchange=0` (la forma normalizada anulaba magnitud $). PBRS `w_tier`. `w_raze_prod` gated a raze **de este step** (no acumulador de episodio / fog leave). |
| Save cash / cheap train mask | `action_adapter.py` | `should_save_for_weap` + `CHEAP_TRAIN_ROLES` (+ `e1`/`e2`). |
| `IDENTITY_ITEMS` + `sample_of` | `roles.py` | Tech/uniques no pasan por `cheapest_of`. |
| Auto-support `weap`/`dome` | `auto_support.py` | PLACE/BUILD asistido con cash ≥ 2500. |
| Spatial extras (11 ch) | `obs_encoding.augment_spatial` | Prod/defensa/infra tipificados, armor/inf, threat. Stem `spatial_extra` zero-init. |
| `SCALAR_DIM=41` | `obs_encoding.py` | Weap-save, colas vehículo, power proyectado, Lanchester, `has_weap`, … |
| Joint entity XF | `network.encode_features` | Buildings → mismo stream que units (`building_tok_proj` + flag). Solo slots **válidos** del batch (no pad fijo a 96). |
| Spatial cross-attn | `network._spatial_cross_attn` | Units Q, fmap 16×16 K/V. |
| Scatter nonzero init | `scatter_proj` | std 0.02 (antes zero-init mataba el acoplamiento). |
| PPO desacoplado | `network` + `trainer` | `lp_macro` (type+item) / `lp_micro` (unit/building/cell); ratios aparte, ventaja compartida. |
| K=2 eco+push | `rollout` + `act_combat` | Aprox. práctica de π_macro·π_micro en el mismo macro-tick (no dos cabezas simultáneas formales). |
| MAB oponentes | `pfsp.mab_probs` | Softmax (1−WR)/τ + floor. |
| Dashboard | `dashboard.html` | Reward: win/raze/mining/**exchange**/**tier**. Pill `shaper_preset`. Sin chart Hyper duplicado. |

### Deliberadamente no 1:1 con el plan largo

| Pedido original | Qué hay |
|-----------------|---------|
| π_macro · π_micro paralelas | K=2 + split de loss |
| Sub-cabeza rol→ítem aprendida | Roles + `sample_of` / `IDENTITY_ITEMS` |
| GTrXL / memoria jerárquica | **Omitido** (pedido explícito) |
| Pipeline renombrado a 3 etapas puras | Labels A…E mapeados a 3 etapas |

---

## Performance (RTX 2070 8 GB)

Síntoma post-arch: `upd` ~6× por sample + spill a **Shared GPU memory**.

Causas: XF joint pad a `MAX_BUILDINGS=96` siempre; `--xf-topk 16` / `--qsa-topk 8` (path custom materializa N×N).

Mitigaciones en `TRAIN_ARGS` (`auto_train.py`):

1. Joint XF solo con buildings válidos (código).
2. **`--xf-topk 0`** y **`--qsa-topk 0`** (dense fused).
3. Si aún hay spill: bajar `--batch-size` (preferir **64** antes que 96).
4. `bptt` al final (pega crédito temporal macro).

Objetivo: dedicated &lt; ~7.5 GB, shared ≈ 0.

---

## Cómo arrancar / retomar

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
# scratch (curriculum nuevo):
.\venv\Scripts\python.exe rl\auto_train.py --scratch --onboard
# resume:
.\venv\Scripts\python.exe rl\auto_train.py --onboard
```

Ckpt dir default onboard: `rl/ckpts_v2`. Dashboard: `dashboard.html?dir=rl/ckpts_v2`.

Tests útiles: `rl/tests/test_macro_first.py`, `rl/tests/test_fog_raze_prod.py`.

---

## Mapa código

| Tema | Archivos |
|------|----------|
| Currícula / promote | `rl/onboard.py`, `rl/auto_train.py` |
| Teacher expand | `rl/scripted_teacher.py` |
| Reward v5 | `rl/reward_shaping.py` |
| Red / PPO split | `rl/network.py`, `rl/trainer.py`, `rl/rollout.py` |
| Obs | `rl/obs_encoding.py` |
| Roles / save weap | `rl/roles.py`, `rl/action_adapter.py`, `rl/auto_support.py` |
| MAB | `rl/pfsp.py` |
