"""Policies for the v3 commander.

A policy picks the commander brain and, for brain "external", answers the
pending decisions of each step.
"""

from typing import Protocol

from ra_env import StepResult, pb


class Policy(Protocol):
    brain: str  # "external" | "builtin"

    def reset(self, seed: int) -> None: ...

    def decide(self, step: StepResult) -> list[pb.Decision]: ...


class BuiltinPolicy:
    """The normal AI's own heuristics drive the commander (executor parity / teacher)."""

    brain = "builtin"

    def reset(self, seed: int) -> None:
        pass

    def decide(self, step: StepResult) -> list[pb.Decision]:
        return []


def make_policy(name: str) -> Policy:
    if name == "builtin":
        return BuiltinPolicy()
    if name == "scripted":
        from ra_rl.policies.scripted import ScriptedPolicy
        return ScriptedPolicy()
    raise ValueError(f"unknown policy {name!r}; expected builtin | scripted")
