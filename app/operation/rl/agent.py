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


def _is_vec_env(env) -> bool:
    from stable_baselines3.common.vec_env import VecEnv

    return isinstance(env, VecEnv)


def _symmetric_actions(venv):
    """Vectorised env whose actions are in [-1, 1] and reach the base env as [0, 1]."""
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class SymmetricActions(VecEnvWrapper):
        def __init__(self, env):
            super().__init__(
                env,
                action_space=spaces.Box(
                    -1.0, 1.0, shape=env.action_space.shape, dtype=np.float32
                ),
            )

        def reset(self):
            return self.venv.reset()

        def step_async(self, actions):
            self.venv.step_async(np.clip((np.asarray(actions) + 1.0) / 2.0, 0.0, 1.0))

        def step_wait(self):
            return self.venv.step_wait()

    return SymmetricActions(venv)


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
        policy_kwargs = {"net_arch": list(s.net_arch)}
        if s.algo == "PPO":
            policy_kwargs["log_std_init"] = s.log_std_init
        kwargs = dict(
            learning_rate=s.learning_rate,
            gamma=s.gamma,
            seed=s.seed,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        if s.algo == "PPO":
            kwargs.update(
                n_steps=s.n_steps, batch_size=s.batch_size, ent_coef=s.ent_coef
            )
        return algo(s.policy, env, **kwargs)

    def prepare(self, env) -> None:
        """Build the model for ``env`` (with symmetric actions if configured)."""
        if self.spec.rescale_actions and _is_vec_env(env):
            env = _symmetric_actions(env)
        self.model = self.build(env)

    def pretrain(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 0,
    ) -> dict:
        """Behaviour cloning of the policy mean and regression of the critic on the returns.

        ``actions`` are in the environment's [0, 1]; ``returns`` the discounted return of every
        observation. Warm start: PPO then refines a policy that already behaves like the rule.
        Needs ``prepare`` (or ``train``) to have built the model.
        """
        import torch

        policy = self.model.policy
        rescaled = bool(
            np.all(self.model.action_space.low == -1.0)
            and np.all(self.model.action_space.high == 1.0)
        )
        if rescaled:
            target = np.clip(actions * 2.0 - 1.0, -0.98, 0.98)
        else:
            target = np.clip(actions, 0.02, 0.98)
        dev = policy.device
        x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        y = torch.as_tensor(target, dtype=torch.float32, device=dev)
        g = torch.as_tensor(returns, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        loss_pi = loss_v = float("nan")
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for i in range(0, len(x), batch_size):
                idx = torch.as_tensor(order[i : i + batch_size], device=dev)
                mean = policy.get_distribution(x[idx]).distribution.mean
                value = policy.predict_values(x[idx]).flatten()
                l_pi = torch.nn.functional.mse_loss(mean, y[idx])
                l_v = torch.nn.functional.mse_loss(value, g[idx])
                opt.zero_grad()
                (l_pi + 0.5 * l_v).backward()
                opt.step()
                loss_pi, loss_v = float(l_pi), float(l_v)
        return dict(policy_mse=loss_pi, value_mse=loss_v, samples=len(x))

    def train(self, env, total_timesteps=None, callback=None, **kwargs):
        if self.model is None:
            self.prepare(env)
        steps = total_timesteps or self.spec.total_timesteps
        log.info(f"training {self.spec.algo} for {steps} steps")
        self.model.learn(total_timesteps=steps, callback=callback, **kwargs)
        return self

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        space = self.model.action_space
        if np.all(space.low == -1.0) and np.all(space.high == 1.0):
            action = (action + 1.0) / 2.0  # trained on rescaled actions
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
