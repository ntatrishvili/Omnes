"""Specification of a reinforcement-learning problem (what the DSL ``RL:`` block describes).

Plain dataclasses, in the style of ``TrafoLimit`` / ``WindowSolution`` of the generic
optimisation: they carry the *description* of the problem (episode, events, action,
observation, reward, agent) and are consumed by ``OmnesEnv`` and ``RLAgent``. They are not
entities of the ``Model``. Build them from Python, from a dict (``RLProblem.from_dict``) or
from a workbook (see the case studies); every value has a default.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Optional

from app.operation.rl.observations import OBSERVATION_REGISTRY
from app.operation.rl.rewards import METRIC_REGISTRY, TRANSFORMS

SOURCES = ("file", "generator", "mixed")
EPISODE_MODES = ("random_day", "sequential", "full_period")
BATTERY_POLICIES = ("self_consumption_rule", "none")
POWER_FLOW_MODES = ("step", "post")
SAFETY_LAYERS = ("none", "laxity")
FORECAST_MODES = ("perfect", "persistence")
REWARD_BASELINES = ("exogenous", "none")


@dataclass
class RewardTerm:
    """One weighted term of the payoff: ``weight * transform(metric)`` (cost, kWh-equivalents).

    :ivar str metric: Name in ``METRIC_REGISTRY``
    :ivar float weight: Non-negative cost per unit of the metric
    :ivar str transform: ``linear`` | ``square`` | ``hinge`` (``max(m - threshold, 0)``)
    :ivar float threshold: Used by ``hinge``
    :ivar dict params: Metric-specific parameters (e.g. ``v_hi`` / ``v_lo``)
    """

    metric: str
    weight: float = 1.0
    transform: str = "linear"
    threshold: float = 0.0
    params: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.metric not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown reward metric '{self.metric}'; known: {sorted(METRIC_REGISTRY)}"
            )
        if self.weight < 0:
            raise ValueError(f"Reward weight of '{self.metric}' must be >= 0")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"Unknown transform '{self.transform}'")


@dataclass
class ObservationTerm:
    """One block of the observation vector (``OBSERVATION_REGISTRY``), with parameters."""

    name: str
    params: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.name not in OBSERVATION_REGISTRY:
            raise ValueError(
                f"Unknown observation term '{self.name}'; known: {sorted(OBSERVATION_REGISTRY)}"
            )


@dataclass
class EpisodeSpec:
    """Episode windows and the train / test split of the days of the data period.

    :ivar int episode_days: Length of one episode in days
    :ivar str mode: ``random_day`` (train days drawn at random) | ``sequential`` | ``full_period``
    :ivar float train_fraction: Share of the days used for training
    :ivar int split_seed: Seed of the day split
    :ivar list train_days: Explicit day indices (override the random split)
    :ivar list test_days: Explicit day indices (override the random split)
    """

    episode_days: int = 1
    mode: str = "random_day"
    train_fraction: float = 0.7
    split_seed: int = 0
    train_days: Optional[list] = None
    test_days: Optional[list] = None


@dataclass
class EventSpec:
    """Where the charging events come from.

    :ivar str source: ``file`` | ``generator`` | ``mixed``
    :ivar str session_file: Charging-session CSV (source ``file`` / ``mixed``)
    :ivar str generator: Name in the event-generator registry (source ``generator`` / ``mixed``)
    :ivar dict generator_params: Keyword arguments of the generator
    :ivar float p_generator: Probability of a generated episode when ``source == mixed``
    """

    source: str = "generator"
    session_file: Optional[str] = None
    generator: str = "dummy"
    generator_params: dict = field(default_factory=dict)
    p_generator: float = 0.5


@dataclass
class ActionSpec:
    """Action = fraction (0..1) of the rated charger power per charger.

    :ivar float charger_kw: Rated power of one charger, if not given by the case study
    :ivar str safety_layer: ``none`` | ``laxity`` (forces the minimum power that still
        delivers the requested energy by the departure)
    """

    charger_kw: Optional[float] = None
    safety_layer: str = "none"


@dataclass
class AgentSpec:
    """Hyper-parameters read by ``BasicPPOAgent`` (and available to subclasses)."""

    algo: str = "PPO"
    policy: str = "MlpPolicy"
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.995
    n_envs: int = 4
    seed: int = 0
    net_arch: list = field(default_factory=lambda: [128, 128])
    n_steps: int = 512
    batch_size: int = 128
    ent_coef: float = 0.0


def default_reward_terms() -> list[RewardTerm]:
    """Recommended payoff: SC and SS first, then feasibility, grid and battery terms."""
    return [
        RewardTerm("import_kwh", 1.0),  # self-sufficiency
        RewardTerm("export_kwh", 1.0),  # self-consumption
        RewardTerm("unmet_kwh", 5.0),  # energy not delivered at departure
        RewardTerm("voltage_excess_pu", 200.0, params={"v_hi": 1.05, "v_lo": 0.95}),
        RewardTerm("trafo_overload_kvah", 3.0),
        RewardTerm("line_overload_kvah", 3.0),
        RewardTerm("battery_throughput_kwh", 0.1),
        RewardTerm("losses_kwh", 0.5),
        RewardTerm("ramp_kw", 0.005),
    ]


def default_observation_terms() -> list[ObservationTerm]:
    return [
        ObservationTerm("time_sin_cos"),
        ObservationTerm("weekday"),
        ObservationTerm("pv_now"),
        ObservationTerm("pv_forecast", {"k": 3}),
        ObservationTerm("base_load_now"),
        ObservationTerm("net_residual"),
        ObservationTerm("battery_soc"),
        ObservationTerm("trafo_loading_prev"),
        ObservationTerm("v_max_prev"),
        ObservationTerm("charger_connected"),
        ObservationTerm("charger_energy_remaining"),
        ObservationTerm("charger_time_remaining"),
        ObservationTerm("charger_laxity"),
        ObservationTerm("charger_p_prev"),
    ]


@dataclass
class RLProblem:
    """Complete description of an RL problem.

    :ivar str id: Name (used for output folders)
    :ivar dict scenario: Case-study scenario values (e.g. ``added_pv_kwp``, ``n_chargers``)
    :ivar str power_flow_mode: ``step`` (power flow in every step, voltage / loading terms
        in the reward) | ``post`` (power flow only when evaluating; much faster training)
    :ivar str battery_policy: ``self_consumption_rule`` | ``none``
    :ivar str forecast_mode: ``perfect`` (read from the data) | ``persistence`` (value of
        the previous day)
    :ivar str reward_baseline: ``exogenous`` subtracts the policy-independent cost of the
        campus without EV and battery (variance reduction) | ``none``
    :ivar float reward_scale: Reference energy in kWh the cost is divided by; ``None`` =
        total rated charger energy per step
    """

    id: str = "rl_problem"
    scenario: dict = field(default_factory=dict)
    power_flow_mode: str = "step"
    battery_policy: str = "self_consumption_rule"
    forecast_mode: str = "perfect"
    reward_baseline: str = "exogenous"
    reward_scale: Optional[float] = None
    episode: EpisodeSpec = field(default_factory=EpisodeSpec)
    events: EventSpec = field(default_factory=EventSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    observation: list = field(default_factory=default_observation_terms)
    reward: list = field(default_factory=default_reward_terms)
    agent: AgentSpec = field(default_factory=AgentSpec)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` for unknown names and impossible values."""
        for name, value, allowed in (
            ("power_flow_mode", self.power_flow_mode, POWER_FLOW_MODES),
            ("battery_policy", self.battery_policy, BATTERY_POLICIES),
            ("forecast_mode", self.forecast_mode, FORECAST_MODES),
            ("reward_baseline", self.reward_baseline, REWARD_BASELINES),
            ("episode.mode", self.episode.mode, EPISODE_MODES),
            ("events.source", self.events.source, SOURCES),
            ("action.safety_layer", self.action.safety_layer, SAFETY_LAYERS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got '{value}'")
        if self.episode.episode_days < 1:
            raise ValueError("episode.episode_days must be >= 1")
        if not 0.0 < self.episode.train_fraction <= 1.0:
            raise ValueError("episode.train_fraction must be in (0, 1]")
        if self.reward_scale is not None and self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        for term in list(self.reward) + list(self.observation):
            term.validate()
        if not self.reward:
            raise ValueError("at least one reward term is required")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "RLProblem":
        """Build a problem from nested dicts / lists (e.g. parsed from a workbook or JSON).

        ``reward`` / ``observation`` are lists of dicts (or term objects); the group keys
        ``episode``, ``events``, ``action``, ``agent`` are dicts. Unknown keys raise.
        """
        cfg = dict(cfg)
        groups = {
            "episode": EpisodeSpec,
            "events": EventSpec,
            "action": ActionSpec,
            "agent": AgentSpec,
        }
        for key, klass in groups.items():
            if key in cfg and isinstance(cfg[key], dict):
                _check_keys(klass, cfg[key], key)
                cfg[key] = klass(**cfg[key])
        if "reward" in cfg:
            cfg["reward"] = [
                t if isinstance(t, RewardTerm) else RewardTerm(**t)
                for t in cfg["reward"]
            ]
        if "observation" in cfg:
            cfg["observation"] = [
                t if isinstance(t, ObservationTerm) else ObservationTerm(**t)
                for t in cfg["observation"]
            ]
        _check_keys(cls, cfg, "problem")
        return cls(**cfg)

    def reward_weights(self) -> dict[str, float]:
        """Weight per metric (sum of the weights if a metric appears twice)."""
        out: dict[str, float] = {}
        for t in self.reward:
            out[t.metric] = out.get(t.metric, 0.0) + t.weight
        return out


def _check_keys(klass, cfg: dict, where: str) -> None:
    unknown = set(cfg) - {f.name for f in fields(klass)}
    if unknown:
        raise ValueError(f"Unknown keys in '{where}': {sorted(unknown)}")
