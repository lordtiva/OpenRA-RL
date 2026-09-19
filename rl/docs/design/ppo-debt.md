# PPO debt (deferred fixes)

Tracked so we do not forget after the cheap audit pack (1A / 2A / 2B / 3B).

## Still open

### 1B — _LOG_RATIO_CLAMP = 2.0 (
l/trainer.py)
Trust-region bypass: allows policy ratios up to ~e^2. Revisit once attack-cell
override is off and remap noise drops; consider lowering clamp or removing it
if spatial_corr.redirect_rate stays near 0.

### 1C — K=2 push reward = 0 (
l/rollout.py)
Macro eco gets the whole block reward; student_push_sample["reward"]=0.
Split or share 
_frame so combat credit reaches the push action.

### 1D — 	imeout_wipe non-Markovian (
l/reward_shaping.py eradicate_v6)
Terminal subtracts cumulative episode return. Prefer a fixed fail penalty
(e.g. -2) without wiping Σr, once turtle+incomplete is under control.

### 2C — Decoupled PPO half-weight on pure macro (
l/trainer.py)
pi_t = -0.5 * (surr_m + surr_u) dilutes train/build when micro is a dummy
ratio=1. Use full surr_m when the action has no micro head.

## Done in the cheap pack
- **1A** --attack-cell-override off (default): no war_objective snap in train.
- **2A** unit_xf_scale init 0.1; load_checkpoint bumps ?0 ? 0.1 so old ckpts enable XF.
- **2B** GradScaler always step then update.
- **3B** MAB Gaussian around WR=0.5.
