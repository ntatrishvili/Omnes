"""Figures of the RL evaluation (static PNGs; style shared with ``plots.py``).

Inputs are the files written by ``train`` / ``evaluate``: ``rl_kpis.csv``, ``episodes.pkl``,
learning curves, sweep / generalisation tables and ``details/<policy>.npz``. Every function
takes its data and an output path and returns the path (or ``None`` if there is nothing to
draw).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.case_studies.kfki_campus.plots import (  # also sets the shared rcParams
    AXIS,
    BLUE,
    BLUE_RAMP,
    CMAP_BLUE,
    GREY,
    INK,
    INK2,
    MUTED,
    ORANGE,
    ORANGE_RAMP,
    plt,
)

# categorical slots, fixed order: learned agents first, then the references
POLICY_COLORS = {
    "uncontrolled": GREY,
    "pv_following": "#2f9e7a",
    "random": "#b1799f",
    "oracle": INK2,
}
AGENT_COLOR = BLUE


def _color(policy: str) -> str:
    return POLICY_COLORS.get(policy, AGENT_COLOR)


def _is_agent(policy: str) -> bool:
    return policy not in POLICY_COLORS


def _save(fig, path: Path) -> Path:
    path = Path(path).resolve()  # absolute: no doubt where the file went
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    if not path.is_file():
        raise RuntimeError(f"figure was not written: {path}")
    return path


def _order(policies) -> list[str]:
    """Learned agents first, then references in a fixed order."""
    ref = ["uncontrolled", "pv_following", "oracle", "random"]
    ag = sorted(p for p in policies if _is_agent(p))
    return ag + [p for p in ref if p in policies]


def _agent_name(kpis: pd.DataFrame) -> Optional[str]:
    agents = [p for p in kpis["policy"].unique() if _is_agent(p)]
    return agents[0] if agents else None


# --------------------------------------------------------------------------- 1 learning curve
def learning_curve(curves: pd.DataFrame, out: Path) -> Optional[Path]:
    """Evaluation reward, SC, SS and unmet energy against training steps (seed band)."""
    if curves is None or curves.empty:
        return None
    panels = [
        ("eval_reward", "Reward per day (test days)"),
        ("SC", "Self-consumption SC"),
        ("SS", "Self-sufficiency SS"),
        ("ev_unmet_kwh", "Unmet EV energy per day (kWh)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (col, title) in zip(axes, panels):
        g = curves.groupby("timesteps")[col]
        mean, lo, hi = g.mean(), g.min(), g.max()
        ax.plot(mean.index, mean.values, color=BLUE, lw=1.8)
        if curves["seed"].nunique() > 1:
            ax.fill_between(
                mean.index, lo.values, hi.values, color=BLUE, alpha=0.18, lw=0
            )
        if col == "eval_reward" and "train_reward" in curves:
            tr = curves.groupby("timesteps")["train_reward"].mean()
            ax.plot(
                tr.index,
                tr.values,
                color=ORANGE,
                lw=1.2,
                ls="--",
                label="training episodes",
            )
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("training steps")
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 2 KPI comparison
KPI_PANELS = [
    ("SC", "SC (-)"),
    ("SS", "SS (-)"),
    ("ev_unmet_kwh", "Unmet EV energy (kWh/day)"),
    ("ev_pv_share", "EV energy in PV surplus (-)"),
    ("v_max_pu", "Max voltage (pu)"),
    ("trafo_max_loading_pct", "Max transformer loading (%)"),
    ("battery_cycles", "Battery cycles per day"),
    ("losses_kwh", "Losses (kWh/day)"),
]


def kpi_comparison(kpis: pd.DataFrame, out: Path) -> Path:
    """Distribution over the test days of the main KPIs for every policy."""
    pol = _order(kpis["policy"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    for ax, (col, title) in zip(axes.ravel(), KPI_PANELS):
        if col not in kpis:
            ax.axis("off")
            continue
        data = [kpis.loc[kpis["policy"] == p, col].dropna().to_numpy() for p in pol]
        bp = ax.boxplot(
            data,
            vert=True,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=INK, lw=1.4),
        )
        for patch, p in zip(bp["boxes"], pol):
            patch.set(facecolor=_color(p), alpha=0.75, edgecolor=AXIS)
        for i, d in enumerate(data, start=1):
            ax.plot(
                i,
                np.mean(d) if len(d) else np.nan,
                marker="D",
                ms=4,
                color=INK,
                zorder=5,
            )
        ax.set_xticks(range(1, len(pol) + 1))
        ax.set_xticklabels([p.replace("_", "\n") for p in pol], fontsize=8)
        ax.set_title(title)
    fig.suptitle(
        "Policies on the test days (box: quartiles, diamond: mean)", color=INK2, y=1.0
    )
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 3 reward decomposition
def reward_decomposition(kpis: pd.DataFrame, out: Path) -> Optional[Path]:
    """Mean reward per day split by term (negative = cost) for every policy."""
    cols = [c for c in kpis if c.startswith("r_") and c != "r_baseline"]
    cols = [c for c in cols if kpis[c].abs().max() > 1e-12]
    if not cols:
        return None
    pol = _order(kpis["policy"].unique())
    m = kpis.groupby("policy")[cols].mean().loc[pol]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ramp = (BLUE_RAMP + ORANGE_RAMP)[: max(len(cols), 1)]
    cmap = plt.get_cmap("tab10")
    bottom = np.zeros(len(pol))
    for i, c in enumerate(cols):
        ax.barh(
            range(len(pol)),
            m[c].to_numpy(),
            left=bottom,
            color=cmap(i),
            label=c[2:],
            height=0.6,
        )
        bottom += m[c].to_numpy()
    ax.plot(
        bottom, range(len(pol)), "D", color=INK, ms=5, label="net (without baseline)"
    )
    ax.set_yticks(range(len(pol)))
    ax.set_yticklabels(pol)
    ax.invert_yaxis()
    ax.set_xlabel("reward contribution per day (normalised, negative = cost)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.set_title("Where each policy loses reward")
    return _save(fig, out)


# --------------------------------------------------------------------------- 4 example days
def _pick_days(episodes: dict, ref_policy: str) -> list[tuple[str, int]]:
    days = sorted({d for (p, d) in episodes if p == ref_policy})
    if not days:
        return []
    pv = {d: episodes[(ref_policy, d)]["dispatch"]["pv"].sum() for d in days}
    ordered = sorted(days, key=pv.get)
    picks = [
        ("high PV", ordered[-1]),
        ("median PV", ordered[len(ordered) // 2]),
        ("low PV", ordered[0]),
    ]
    seen, out = set(), []
    for label, d in picks:
        if d not in seen:
            seen.add(d)
            out.append((label, d))
    return out


def example_days(episodes: dict, policies: list[str], out: Path) -> Optional[Path]:
    """EV power (stacked) and PV surplus over the rest of the campus on three contrasting days."""
    ref = policies[0]
    picks = _pick_days(episodes, ref)
    if not picks:
        return None
    fig, axes = plt.subplots(
        len(policies),
        len(picks),
        figsize=(4.8 * len(picks), 2.9 * len(policies)),
        squeeze=False,
    )
    for c, (label, day) in enumerate(picks):
        for r, pol in enumerate(policies):
            ax = axes[r][c]
            ep = episodes.get((pol, day))
            if ep is None:
                ax.axis("off")
                continue
            disp, ev = ep["dispatch"], ep["ev_power"]
            hours = (disp.index - disp.index[0]) / pd.Timedelta("1h")
            base = disp["demand"] - ev.sum(axis=1)
            ax.stackplot(
                hours,
                ev.to_numpy().T,
                colors=[
                    CMAP_BLUE(0.25 + 0.7 * i / max(ev.shape[1] - 1, 1))
                    for i in range(ev.shape[1])
                ],
                alpha=0.9,
                linewidth=0,
            )
            ax2 = (
                ax.twinx()
            )  # campus scale (MW) on the right, EV scale (kW) on the left
            ax2.plot(
                hours,
                (disp["pv"] - base).clip(lower=0),
                color=ORANGE,
                lw=1.6,
                label="PV surplus over campus load (right axis)",
            )
            ax2.plot(
                hours,
                disp["p_export"],
                color=INK2,
                lw=1.0,
                ls=":",
                label="grid export (right axis)",
            )
            ax2.grid(False)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color(AXIS)
            ax2.set_ylim(bottom=0)
            ax.set_ylim(0, max(float(ev.to_numpy().sum(axis=1).max()) * 1.2, 5.0))
            ax.set_xlim(0, 24)
            if r == 0:
                ax.set_title(f"{label} - {disp.index[0]:%Y-%m-%d}")
            if c == 0:
                ax.set_ylabel(f"{pol}\nEV power (kW)")
            if r == len(policies) - 1:
                ax.set_xlabel("hour")
            if r == 0 and c == 0:
                ax2.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "EV charging power (stacked areas, one per charger) against the PV surplus",
        y=1.0,
        color=INK2,
    )
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 5 carpet + gantt
def ev_carpet(episodes: dict, policies: list[str], out: Path) -> Optional[Path]:
    """Aggregate EV power by hour of day (x) and test day (y) for each policy."""
    mats = {}
    for pol in policies:
        days = sorted(d for (p, d) in episodes if p == pol)
        if not days:
            continue
        rows = []
        for d in days:
            s = episodes[(pol, d)]["ev_power"].sum(axis=1).to_numpy()
            rows.append(np.pad(s[:24], (0, max(0, 24 - len(s)))))
        mats[pol] = np.array(rows)
    if not mats:
        return None
    vmax = max(m.max() for m in mats.values()) or 1.0
    fig, axes = plt.subplots(
        1, len(mats), figsize=(3.6 * len(mats) + 1.2, 4.2), sharey=True, squeeze=False
    )
    for ax, (pol, m) in zip(axes[0], mats.items()):
        im = ax.imshow(
            m,
            aspect="auto",
            cmap=CMAP_BLUE,
            vmin=0,
            vmax=vmax,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(pol)
        ax.set_xlabel("hour of day")
        ax.grid(False)
    axes[0][0].set_ylabel("test day (sorted)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="EV power (kW)", shrink=0.8)
    return _save(fig, out)


def session_gantt(
    episodes: dict, policies: list[str], day: int, out: Path
) -> Optional[Path]:
    """Sessions of one day per charger: plug-in to plug-out, coloured by the charging power."""
    have = [p for p in policies if (p, day) in episodes]
    if not have:
        return None
    fig, axes = plt.subplots(
        len(have), 1, figsize=(10, 2.4 * len(have) + 0.6), sharex=True, squeeze=False
    )
    vmax = max(episodes[(p, day)]["ev_power"].to_numpy().max() for p in have) or 1.0
    for ax, pol in zip(axes[:, 0], have):
        ep = episodes[(pol, day)]
        ev, sess = ep["ev_power"], ep["sessions"]
        t0 = ev.index[0]
        for j, cid in enumerate(ev.columns):
            for i, (ts, p) in enumerate(ev[cid].items()):
                h0 = (ts - t0) / pd.Timedelta("1h")
                if p > 1e-6:
                    ax.barh(
                        j,
                        1.0,
                        left=h0,
                        color=CMAP_BLUE(0.25 + 0.75 * p / vmax),
                        height=0.7,
                        linewidth=0,
                    )
            if len(sess):
                for s in sess[sess["charger_id"] == cid].itertuples():
                    a = (s.arrival - t0) / pd.Timedelta("1h")
                    d = (s.departure - t0) / pd.Timedelta("1h")
                    ax.barh(
                        j,
                        d - a,
                        left=a,
                        height=0.8,
                        fill=False,
                        edgecolor=MUTED,
                        lw=0.8,
                    )
        ax.set_yticks(range(len(ev.columns)))
        ax.set_yticklabels(list(ev.columns), fontsize=8)
        ax.set_title(f"{pol} - {t0:%Y-%m-%d}", loc="left")
        ax.set_xlim(0, 24)
    axes[-1, 0].set_xlabel("hour")
    return _save(fig, out)


# --------------------------------------------------------------------------- 6 satisfaction
def session_satisfaction(
    episodes: dict, policies: list[str], out: Path
) -> Optional[Path]:
    """Delivered / requested energy per session (histogram) and CDF of the unmet energy."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    any_data = False
    for pol in policies:
        frames = [
            ep["sessions"]
            for (p, _), ep in episodes.items()
            if p == pol and len(ep["sessions"])
        ]
        if not frames:
            continue
        s = pd.concat(frames)
        any_data = True
        frac = (s["delivered_kwh"] / s["requested_kwh"]).clip(0, 1)
        a1.hist(
            frac,
            bins=np.linspace(0, 1, 21),
            histtype="step",
            color=_color(pol),
            lw=1.6,
            label=pol,
        )
        unmet = np.sort(s["requested_kwh"] - s["delivered_kwh"])
        a2.plot(
            unmet,
            np.arange(1, len(unmet) + 1) / len(unmet),
            color=_color(pol),
            lw=1.6,
            label=pol,
        )
    if not any_data:
        plt.close(fig)
        return None
    a1.set_xlabel("delivered / requested energy per session")
    a1.set_ylabel("sessions")
    a1.set_yscale("log")
    a2.set_xlabel("unmet energy per session (kWh)")
    a2.set_ylabel("cumulative share of sessions")
    a2.set_xscale("symlog", linthresh=0.5)
    a1.legend(fontsize=8)
    return _save(fig, out)


# --------------------------------------------------------------------------- 7 grid side
def grid_duration(episodes: dict, policies: list[str], out: Path) -> Optional[Path]:
    """Duration curves of the highest bus voltage and of the highest transformer loading."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    any_data = False
    for pol in policies:
        eps = [ep for (p, _), ep in episodes.items() if p == pol]
        if not eps:
            continue
        vm = np.concatenate([np.nanmax(e["vm_pu"], axis=1) for e in eps])
        tl = np.concatenate([np.nanmax(e["trafo_loading"], axis=1) for e in eps])
        vm, tl = vm[~np.isnan(vm)], tl[~np.isnan(tl)]
        if not len(vm):
            continue
        any_data = True
        x = np.linspace(0, 100, len(vm))
        a1.plot(
            np.linspace(0, 100, len(vm)),
            np.sort(vm)[::-1],
            color=_color(pol),
            lw=1.6,
            label=pol,
        )
        a2.plot(
            np.linspace(0, 100, len(tl)),
            np.sort(tl)[::-1],
            color=_color(pol),
            lw=1.6,
            label=pol,
        )
    if not any_data:
        plt.close(fig)
        return None
    a1.axhline(1.05, color=ORANGE, lw=0.9, ls="--")
    a1.axhline(1.10, color=ORANGE, lw=0.9)
    a1.set_ylabel("highest bus voltage (pu)")
    a2.axhline(100, color=ORANGE, lw=0.9)
    a2.set_ylabel("highest transformer loading (%)")
    for ax in (a1, a2):
        ax.set_xlabel("share of the time (%)")
    a1.legend(fontsize=8)
    return _save(fig, out)


# --------------------------------------------------------------------------- 8 pareto / sweep
def sweep_pareto(sweep: pd.DataFrame, out: Path) -> Optional[Path]:
    """Trade-off along a reward weight: SC / SS against unmet energy and voltage violations."""
    if sweep is None or sweep.empty:
        return None
    metric = sweep["metric"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    pairs = [("ev_unmet_kwh", "SS"), ("ev_unmet_kwh", "SC"), ("viol_frac", "SS")]
    for ax, (x, y) in zip(axes, pairs):
        if x not in sweep or y not in sweep:
            ax.axis("off")
            continue
        sc = ax.scatter(
            sweep[x],
            sweep[y],
            c=np.log10(sweep["weight"].clip(lower=1e-9)),
            cmap=CMAP_BLUE,
            s=60,
            edgecolor=INK2,
        )
        ax.plot(sweep[x], sweep[y], color=AXIS, lw=0.8, zorder=0)
        for r in sweep.itertuples():
            ax.annotate(
                f"{r.weight:g}",
                (getattr(r, x), getattr(r, y)),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    fig.suptitle(
        f"Sensitivity to the weight of '{metric}' (labels: weight)", y=1.0, color=INK2
    )
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 9 generalisation
def generalization_matrix(df: pd.DataFrame, out: Path) -> Optional[Path]:
    """Train source (rows) x test source (columns): reward, SC, SS and unmet energy."""
    if df is None or df.empty:
        return None
    metrics = [
        ("reward", "reward/day"),
        ("SC", "SC"),
        ("SS", "SS"),
        ("ev_unmet_kwh", "unmet kWh/day"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2))
    for ax, (col, title) in zip(axes, metrics):
        m = df.pivot(index="train_source", columns="test_source", values=col)
        im = ax.imshow(m.to_numpy(), cmap=CMAP_BLUE, aspect="auto")
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(m.columns)
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(m.index)
        ax.grid(False)
        lo, hi = np.nanmin(m.to_numpy()), np.nanmax(m.to_numpy())
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.3g}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if (v - lo) / (hi - lo + 1e-12) > 0.55 else INK,
                )
        ax.set_title(title)
        ax.set_xlabel("test events")
    axes[0].set_ylabel("training events")
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 10 diagnostics
def policy_diagnostics(
    episodes: dict, policies: list[str], out: Path, rated: Optional[np.ndarray] = None
) -> Optional[Path]:
    """Share of the connected charger power used against the PV surplus share (per step)."""
    fig, axes = plt.subplots(
        1, len(policies), figsize=(4.2 * len(policies), 3.8), sharey=True, squeeze=False
    )
    any_data = False
    for ax, pol in zip(axes[0], policies):
        xs, ys, hs = [], [], []
        for (p, _), ep in episodes.items():
            if p != pol:
                continue
            ev, con = ep["ev_power"], ep["connected"]
            r = np.full(con.shape[1], 22.0) if rated is None else rated
            cap = (con * r).sum(axis=1)
            disp = ep["dispatch"]
            base = disp["demand"].to_numpy() - ev.sum(axis=1).to_numpy()
            surplus = disp["pv"].to_numpy() - base
            ok = cap > 0
            xs.append(np.clip(surplus[ok] / cap[ok], -1, 3))
            ys.append(ev.sum(axis=1).to_numpy()[ok] / cap[ok])
            hs.append(disp.index.hour.to_numpy()[ok])
        if not xs:
            ax.axis("off")
            continue
        any_data = True
        x, y, h = np.concatenate(xs), np.concatenate(ys), np.concatenate(hs)
        sc = ax.scatter(x, y, c=h, cmap="viridis", s=10, alpha=0.6, linewidths=0)
        ax.set_title(pol)
        ax.set_xlabel("PV surplus / connected charger power")
    if not any_data:
        plt.close(fig)
        return None
    axes[0][0].set_ylabel("charging power / connected charger power")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="hour of day", shrink=0.8)
    return _save(fig, out)


# --------------------------------------------------------------------------- 11 seeds
def seed_variability(kpis: pd.DataFrame, out: Path) -> Optional[Path]:
    """Reward, SC and SS of every trained seed (mean and spread over the test days)."""
    agents = [p for p in kpis["policy"].unique() if _is_agent(p)]
    if len(agents) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col in zip(axes, ("reward", "SC", "SS")):
        data = [kpis.loc[kpis["policy"] == a, col].to_numpy() for a in sorted(agents)]
        bp = ax.boxplot(
            data,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=INK),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=BLUE, alpha=0.6, edgecolor=AXIS)
        ref = kpis.loc[kpis["policy"] == "uncontrolled", col].mean()
        if not np.isnan(ref):
            ax.axhline(ref, color=GREY, lw=1.2, ls="--", label="uncontrolled (mean)")
        ax.set_xticklabels(
            [a.replace("_seed", "\nseed ") for a in sorted(agents)], fontsize=8
        )
        ax.set_title(col)
    axes[0].legend(fontsize=8)
    return _save(fig, out)


# --------------------------------------------------------------------------- driver
def make_rl_figures(out_dir: Path) -> list[Path]:
    """Draw every figure for which the result files exist in ``out_dir``."""
    import pickle

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    made: list[Optional[Path]] = []
    kp = out_dir / "rl_kpis.csv"
    curves = [pd.read_csv(f) for f in sorted((out_dir / "models").glob("*_curve.csv"))]
    if curves:
        made.append(
            learning_curve(pd.concat(curves), fig_dir / "rl_learning_curve.png")
        )
    if kp.exists():
        kpis = pd.read_csv(kp)
        pol = _order(kpis["policy"].unique())
        main = [p for p in pol if p in ("uncontrolled", "pv_following", "oracle")]
        agent = _agent_name(kpis)
        view = ([agent] if agent else []) + main
        made.append(kpi_comparison(kpis, fig_dir / "rl_kpi_comparison.png"))
        made.append(reward_decomposition(kpis, fig_dir / "rl_reward_decomposition.png"))
        made.append(seed_variability(kpis, fig_dir / "rl_seed_variability.png"))
        ep_file = out_dir / "episodes.pkl"
        if ep_file.exists():
            with open(ep_file, "rb") as f:
                episodes = pickle.load(f)
            cmp = ([agent] if agent else []) + [
                p for p in ("uncontrolled", "pv_following") if p in pol
            ]
            made.append(
                example_days(episodes, cmp[:3], fig_dir / "rl_example_days.png")
            )
            made.append(ev_carpet(episodes, view, fig_dir / "rl_ev_carpet.png"))
            days = sorted({d for (_, d) in episodes})
            if days:
                made.append(
                    session_gantt(
                        episodes,
                        cmp[:3],
                        days[len(days) // 2],
                        fig_dir / "rl_session_gantt.png",
                    )
                )
            made.append(
                session_satisfaction(
                    episodes, view, fig_dir / "rl_session_satisfaction.png"
                )
            )
            made.append(grid_duration(episodes, view, fig_dir / "rl_grid_duration.png"))
            made.append(
                policy_diagnostics(
                    episodes, cmp[:3], fig_dir / "rl_policy_diagnostics.png"
                )
            )
    if (out_dir / "rl_sweep.csv").exists():
        made.append(
            sweep_pareto(
                pd.read_csv(out_dir / "rl_sweep.csv"), fig_dir / "rl_sweep_pareto.png"
            )
        )
    if (out_dir / "rl_generalization.csv").exists():
        made.append(
            generalization_matrix(
                pd.read_csv(out_dir / "rl_generalization.csv"),
                fig_dir / "rl_generalization.png",
            )
        )
    return [m for m in made if m is not None]
