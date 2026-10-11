"""Generic Gymnasium environment on top of an OMNES ``Model``.

``OmnesEnv`` is to RL what ``EnergySystemOptimizer`` is to optimisation: it fixes the
procedure (day split, episode windows, event sampling, reward, observation vector, spaces)
and leaves the physics to a subclass of the case study:

* ``_setup()``                 build everything that depends on the model (once)
* ``_start_episode(...)``      initial state of an episode window
* ``_physics_step(a)``         apply the action, advance one step, return the reward context
* ``_observe_state()``         the ``ObsState`` the observation terms read

One environment step is one time step of the model.
"""

from abc import abstractmethod
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from app.infra.logging_setup import get_logger
from app.operation.rl.events import (
    GENERATOR_REGISTRY,
    ChargingEvent,
    ChargingEventGenerator,
)
from app.operation.rl.observations import (
    ObsState,
    build_observation,
    observation_size,
)
from app.operation.rl.problem import RLProblem
from app.operation.rl.rewards import RewardFunction

log = get_logger(__name__)


def split_days(
    n_days: int, episode_days: int, fraction: float, seed: int, train=None, test=None
) -> tuple[list[int], list[int]]:
    """Train / test start days. Explicit lists win; otherwise a seeded random split.

    A start day is valid when a whole episode fits behind it. With ``fraction == 1`` the test
    days are the training days.
    """
    valid = list(range(0, max(n_days - episode_days + 1, 1)))
    if train is not None or test is not None:
        tr = [d for d in (train if train is not None else valid) if d in valid]
        te = [d for d in (test if test is not None else valid) if d in valid]
        return sorted(tr), sorted(te)
    perm = np.random.default_rng(seed).permutation(valid)
    n_train = max(1, int(round(fraction * len(valid))))
    tr, te = sorted(perm[:n_train].tolist()), sorted(perm[n_train:].tolist())
    return tr, (te or tr)


class OmnesEnv(gym.Env):
    """Base environment; see the module docstring for the hooks.

    :param RLProblem problem: The problem description
    :param str split: ``train`` or ``test`` days are drawn / iterated
    :ivar RewardFunction reward_fn: Payoff built from ``problem.reward``
    :ivar list chargers: Ids of the controlled chargers (the action has one entry per charger)
    """

    metadata = {"render_modes": []}

    def __init__(self, problem: RLProblem, split: str = "train"):
        super().__init__()
        self.problem = problem
        self.split = split
        self.chargers: dict[str, float] = {}
        self.dt_h = 1.0
        self.steps_per_day = 24
        self.n_steps_total = 0
        self.index: Optional[pd.DatetimeIndex] = None
        self._file_generator: Optional[ChargingEventGenerator] = None
        self._setup()
        n = len(self.chargers)
        self.action_space = spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)
        size = observation_size(problem.observation, n)
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(size,), dtype=np.float32
        )
        n_days = self.n_steps_total // self.steps_per_day
        self.train_days, self.test_days = split_days(
            n_days,
            problem.episode.episode_days,
            problem.episode.train_fraction,
            problem.episode.split_seed,
            problem.episode.train_days,
            problem.episode.test_days,
        )
        e_ref = problem.reward_scale or sum(self.chargers.values()) * self.dt_h
        self.reward_fn = RewardFunction(problem.reward, e_ref, problem.reward_baseline)
        self.generator = self._build_generator()
        self._seq = 0
        self.t0 = 0  # first step of the current window (absolute)
        self.T = 0  # length of the current window
        self.k = 0  # step within the window
        self.day = 0

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _setup(self) -> None:
        """Fill ``chargers``, ``dt_h``, ``steps_per_day``, ``n_steps_total``, ``index``."""

    @abstractmethod
    def _start_episode(self, t0: int, T: int, events: list[ChargingEvent]) -> None:
        """Reset the state for the window ``[t0, t0 + T)`` with the given events."""

    @abstractmethod
    def _physics_step(self, action: np.ndarray) -> tuple[dict, tuple[float, float]]:
        """Advance one step; return the reward context and the (import, export) reference."""

    @abstractmethod
    def _observe_state(self) -> ObsState:
        """State the observation terms read."""

    # ------------------------------------------------------------------ events
    def _build_generator(self) -> Optional[ChargingEventGenerator]:
        ev = self.problem.events
        if ev.source == "file" and self._file_generator is None:
            raise ValueError("events.source 'file' needs a session file")
        if ev.source == "file":
            return None
        if ev.generator not in GENERATOR_REGISTRY:
            raise ValueError(
                f"Unknown event generator '{ev.generator}'; known: {sorted(GENERATOR_REGISTRY)}"
            )
        return GENERATOR_REGISTRY[ev.generator](**ev.generator_params)

    def _events_for(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[ChargingEvent]:
        ev = self.problem.events
        use_generator = ev.source == "generator" or (
            ev.source == "mixed"
            and (
                self._file_generator is None or self.np_random.random() < ev.p_generator
            )
        )
        gen = self.generator if use_generator else self._file_generator
        return gen.sample(self.np_random, start, end, self.chargers)

    # ------------------------------------------------------------------ gym API
    def _pick_day(self, options: dict) -> int:
        days = self.train_days if self.split == "train" else self.test_days
        if "day" in options:
            return int(options["day"])
        mode = self.problem.episode.mode
        if mode == "sequential":
            day = days[self._seq % len(days)]
            self._seq += 1
            return day
        return int(self.np_random.choice(days))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        ep = self.problem.episode
        if ep.mode == "full_period" or options.get("full_period"):
            self.day, self.t0, self.T = 0, 0, self.n_steps_total
        else:
            self.day = self._pick_day(options)
            self.t0 = self.day * self.steps_per_day
            self.T = min(
                ep.episode_days * self.steps_per_day, self.n_steps_total - self.t0
            )
        self.k = 0
        start = self.index[self.t0]
        end = start + pd.Timedelta(hours=self.T * self.dt_h)
        self._start_episode(self.t0, self.T, self._events_for(start, end))
        return self._observation(), {"day": self.day, "start": start}

    def _observation(self) -> np.ndarray:
        return build_observation(self.problem.observation, self._observe_state())

    def step(self, action: Any):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), 0.0, 1.0)
        ctx, ref = self._physics_step(a)
        reward, breakdown = self.reward_fn(ctx, ref)
        self.k += 1
        truncated = self.k >= self.T
        obs = (
            self._observation()
        )  # after the last step: state at the window edge (clamped)
        info = {
            "metrics": self.reward_fn.metrics(ctx),
            "reward_terms": breakdown,
            "import_kwh": ctx["import_kwh"],
            "export_kwh": ctx["export_kwh"],
            "unmet_kwh": ctx["unmet_kwh"],
        }
        return obs, reward, False, truncated, info
