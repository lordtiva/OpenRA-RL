"""Policies that act through the network's action space (features.py)."""

import random

import numpy as np

from ra_env import StepResult, pb
from ra_rl.features import Features
from ra_rl.vocab import QUEUE_INDEX


class RandomPolicy:
    """Uniform over the valid actions of each request (the floor for every criterion)."""

    brain = "external"

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.feats = Features()

    def decide(self, step: StepResult) -> list[pb.Decision]:
        out = []
        for req in step.pending:
            mask = self.feats.candidates(req)[1] if req.kind == pb.DECIDE_ARMY else self.feats.item_mask(req)
            d = self.feats.decision(req, self.rng.choice(np.flatnonzero(mask).tolist()))
            if d is not None:
                out.append(d)
        return out


class NetPolicy:
    """The commander network, greedy (argmax) by default, on CPU."""

    brain = "external"

    def __init__(self, path: str, sample: bool = False):
        import torch

        from ra_rl.model import CommanderNet

        torch.set_num_threads(1)  # one game per thread; tiny batches
        self.torch = torch
        self.net = CommanderNet.load(path)
        self.sample = sample
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.feats = Features()
        self.gen = self.torch.Generator().manual_seed(seed)

    def actions(self, step: StepResult) -> list[int]:
        """Observe the step and pick an action for each pending request (not yet recorded)."""
        torch = self.torch
        self.feats.observe(step.obs)
        if not step.pending:
            return []
        g = torch.from_numpy(self.feats.g).unsqueeze(0)
        out = []
        with torch.no_grad():
            for req in step.pending:
                if req.kind == pb.DECIDE_ARMY:
                    cand, mask = self.feats.candidates(req)
                    logits = self.net.army_logits(g, torch.from_numpy(cand).unsqueeze(0),
                                                  torch.from_numpy(mask).unsqueeze(0))
                else:
                    queue = torch.tensor([QUEUE_INDEX.get(req.queue, 0)])
                    logits = self.net.item_logits(g, queue, torch.from_numpy(self.feats.item_mask(req)).unsqueeze(0))
                if self.sample:
                    a = int(torch.multinomial(torch.softmax(logits[0], -1), 1, generator=self.gen))
                else:
                    a = int(logits[0].argmax())
                out.append(a)
        return out

    def decide(self, step: StepResult) -> list[pb.Decision]:
        out = []
        for req, a in zip(step.pending, self.actions(step), strict=True):
            d = self.feats.decision(req, a)
            self.feats.record(req, d)
            if d is not None:
                out.append(d)
        return out
