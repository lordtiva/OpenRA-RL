import numpy as np
import pytest
import torch

from ra_rl.features import CAND_DIM, GLOBAL_DIM, MAX_TARGETS, N_ITEM_ACTIONS
from ra_rl.model import CommanderNet


def compute_gae(vals: np.ndarray, rewards: np.ndarray, gamma: float, gae_lambda: float):
    T = len(vals)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        reward = rewards[t]
        next_val = 0.0 if t == T - 1 else vals[t + 1]
        delta = reward + gamma * next_val - vals[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + vals[t]
    return advantages, returns


def test_gae_simple_case():
    # T=2, terminal win reward = 1.0, non-terminal = 0.0
    vals = np.array([0.5, 0.8], dtype=np.float32)
    rewards = np.array([0.0, 1.0], dtype=np.float32)
    gamma = 0.9
    lam = 0.95

    adv, ret = compute_gae(vals, rewards, gamma, lam)

    # t=1: delta_1 = 1.0 + 0 - 0.8 = 0.2
    #      adv[1] = 0.2
    #      ret[1] = 0.2 + 0.8 = 1.0
    assert pytest.approx(adv[1], rel=1e-5) == 0.2
    assert pytest.approx(ret[1], rel=1e-5) == 1.0

    # t=0: delta_0 = 0.0 + 0.9 * 0.8 - 0.5 = 0.72 - 0.5 = 0.22
    #      adv[0] = 0.22 + 0.9 * 0.95 * 0.2 = 0.22 + 0.171 = 0.391
    #      ret[0] = adv[0] + vals[0] = 0.391 + 0.5 = 0.891
    assert pytest.approx(adv[0], rel=1e-5) == 0.391
    assert pytest.approx(ret[0], rel=1e-5) == 0.891


def test_ppo_loss_backward():
    net = CommanderNet(hidden=64, cand_hidden=32)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    B_prod = 8
    B_army = 4

    g_prod = torch.randn(B_prod, GLOBAL_DIM)
    queue = torch.randint(0, 6, (B_prod,))
    mask_prod = torch.ones(B_prod, N_ITEM_ACTIONS, dtype=torch.bool)
    act_prod = torch.randint(0, N_ITEM_ACTIONS, (B_prod,))
    old_lp_prod = torch.randn(B_prod)
    adv_prod = torch.randn(B_prod)
    ret_prod = torch.randn(B_prod)

    g_army = torch.randn(B_army, GLOBAL_DIM)
    cand_army = torch.randn(B_army, MAX_TARGETS, CAND_DIM)
    mask_army = torch.ones(B_army, MAX_TARGETS + 2, dtype=torch.bool)
    act_army = torch.randint(0, MAX_TARGETS + 2, (B_army,))
    old_lp_army = torch.randn(B_army)
    adv_army = torch.randn(B_army)
    ret_army = torch.randn(B_army)

    # Forward prod
    logits_prod = net.item_logits(g_prod, queue, mask_prod)
    dist_prod = torch.distributions.Categorical(logits=logits_prod)
    new_lp_prod = dist_prod.log_prob(act_prod)
    ratio_prod = torch.exp(new_lp_prod - old_lp_prod)
    surr1_p = ratio_prod * adv_prod
    surr2_p = torch.clamp(ratio_prod, 0.8, 1.2) * adv_prod
    pi_loss_prod = -torch.min(surr1_p, surr2_p).mean()
    val_loss_prod = torch.nn.functional.mse_loss(net.value(g_prod), ret_prod)

    # Forward army
    logits_army = net.army_logits(g_army, cand_army, mask_army)
    dist_army = torch.distributions.Categorical(logits=logits_army)
    new_lp_army = dist_army.log_prob(act_army)
    ratio_army = torch.exp(new_lp_army - old_lp_army)
    surr1_a = ratio_army * adv_army
    surr2_a = torch.clamp(ratio_army, 0.8, 1.2) * adv_army
    pi_loss_army = -torch.min(surr1_a, surr2_a).mean()
    val_loss_army = torch.nn.functional.mse_loss(net.value(g_army), ret_army)

    loss = pi_loss_prod + pi_loss_army + 0.5 * (val_loss_prod + val_loss_army)
    optimizer.zero_grad()
    loss.backward()

    # Check that all modules got gradients
    assert net.trunk[0].weight.grad is not None
    assert net.value_head[1].weight.grad is not None
    assert net.item_head[1].weight.grad is not None
    assert net.cand_enc[0].weight.grad is not None
    assert net.cand_score[0].weight.grad is not None
    assert net.army_fixed.weight.grad is not None

    optimizer.step()
