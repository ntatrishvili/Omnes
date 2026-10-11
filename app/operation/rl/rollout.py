"""Run a policy on an environment and collect the episode totals."""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from app.operation.rl.agent import RLAgent


@dataclass
class EpisodeResult:
    """Totals of one episode.

    :ivar int day: Start day of the window
    :ivar float reward: Sum of the step rewards
    :ivar dict terms: Reward contribution per term (sum over the episode, incl. ``baseline``)
    :ivar dict metrics: Raw metric per term (sum over the episode)
    :ivar int steps: Number of steps
    """

    day: int
    reward: float
    terms: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    steps: int = 0


def run_episode(
    env,
    agent: RLAgent,
    options: Optional[dict] = None,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> EpisodeResult:
    """One episode of ``agent`` on ``env``; the env keeps its recorded history afterwards."""
    agent.bind(env)
    obs, info = env.reset(seed=seed, options=options)
    total, terms, metrics, steps = 0.0, {}, {}, 0
    done = False
    while not done:
        action = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        for k, v in info["reward_terms"].items():
            terms[k] = terms.get(k, 0.0) + v
        for k, v in info["metrics"].items():
            metrics[k] = metrics.get(k, 0.0) + v
        steps += 1
        done = terminated or truncated
    return EpisodeResult(env.day, total, terms, metrics, steps)


def evaluate_agent(
    env,
    agent: RLAgent,
    days: Optional[Sequence[int]] = None,
    seed: int = 0,
    deterministic: bool = True,
) -> list[EpisodeResult]:
    """Run the agent on the given start days (default: the env's test days).

    Episode ``i`` is seeded with ``seed + i`` so that all policies see the same events.
    """
    days = list(env.test_days if days is None else days)
    return [
        run_episode(env, agent, {"day": d}, seed + i, deterministic)
        for i, d in enumerate(days)
    ]


def mean_reward(results: Sequence[EpisodeResult]) -> float:
    return float(np.mean([r.reward for r in results])) if results else float("nan")
