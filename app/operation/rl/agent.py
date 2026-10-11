"""RL agents: a small interface plus a basic PPO implementation.

``RLAgent`` is deliberately thin so that it can be specialised: override ``build`` to change
the algorithm or the network, or register a new class in ``AGENT_REGISTRY`` and name it in
``AgentSpec.algo``. Stable-Baselines3 is an optional dependency (``requirements-rl.txt``) and is
only imported when an SB3 agent is built.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from app.infra.logging_setup import get_logger
from app.operation.rl.problem import AgentSpec, RLProblem

log = get_logger(__name__)


class RLAgent(ABC):
    """Interface used by training, evaluation and the baselines.

    :ivar AgentSpec spec: Hyper-parameters from the problem description
    """

    name = "agent"

    def __init__(self, problem: Optional[RLProblem] = None):
        self.problem = problem
        self.spec: AgentSpec = problem.agent if problem is not None else AgentSpec()

    def bind(self, env) -> None:
        """Called by the rollout before an episode; rule-based agents read the env here."""

    def train(self, env, total_timesteps: Optional[int] = None, **kwargs) -> "RLAgent":
        """Learn on ``env``. Non-learning agents ignore it."""
        return self

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Action (fractions of rated power per charger) for an observation."""

    def save(self, path: Path) -> None:
        """Persist the agent (no-op for agents without parameters)."""

    @classmethod
    def load(cls, path: Path, problem: Optional[RLProblem] = None) -> "RLAgent":
        return cls(problem)


class RandomAgent(RLAgent):
    """Uniform random setpoints (lower bound of the comparison)."""

    name = "random"

    def __init__(self, problem=None, seed: int = 0):
        super().__init__(problem)
        self._rng = np.random.default_rng(seed)
        self._shape = None

    def bind(self, env) -> None:
        self._shape = env.action_space.shape

    def predict(self, obs, deterministic=True):
        return self._rng.random(self._shape)


def _import_sb3():
    try:
        import stable_baselines3 as sb3
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Stable-Baselines3 is not installed: pip install -r requirements-rl.txt"
        ) from exc
    return sb3


class BasicPPOAgent(RLAgent):
    """Stable-Baselines3 agent configured by ``AgentSpec`` (PPO by default)."""

    name = "ppo"

    def __init__(self, problem=None):
        super().__init__(problem)
        self.model = None

    def build(self, env):
        """Create the SB3 model for ``env``; override to specialise."""
        sb3 = _import_sb3()
        s = self.spec
        algo = getattr(sb3, s.algo)
        kwargs = dict(
            learning_rate=s.learning_rate,
            gamma=s.gamma,
            seed=s.seed,
            policy_kwargs={"net_arch": list(s.net_arch)},
            verbose=0,
        )
        if s.algo == "PPO":
            kwargs.update(
                n_steps=s.n_steps, batch_size=s.batch_size, ent_coef=s.ent_coef
            )
        return algo(s.policy, env, **kwargs)

    def train(self, env, total_timesteps=None, callback=None, **kwargs):
        if self.model is None:
            self.model = self.build(env)
        steps = total_timesteps or self.spec.total_timesteps
        log.info(f"training {self.spec.algo} for {steps} steps")
        self.model.learn(total_timesteps=steps, callback=callback, **kwargs)
        return self

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: Path) -> None:
        self.model.save(str(path))

    @classmethod
    def load(cls, path: Path, problem=None):
        sb3 = _import_sb3()
        agent = cls(problem)
        agent.model = getattr(sb3, agent.spec.algo).load(str(path))
        return agent


AGENT_REGISTRY: dict[str, type[RLAgent]] = {
    "PPO": BasicPPOAgent,
    "SAC": BasicPPOAgent,
    "TD3": BasicPPOAgent,
    "random": RandomAgent,
}


def make_agent(problem: RLProblem) -> RLAgent:
    """Agent class named in ``problem.agent.algo``."""
    if problem.agent.algo not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{problem.agent.algo}'; known: {sorted(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[problem.agent.algo](problem)
