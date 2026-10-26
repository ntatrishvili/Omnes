"""Hungarian figures of the RL case study (PNG 300 dpi and vector PDF, A4-column sized).

Every function takes the tables written by ``run.py`` and an output folder and returns the
paths it wrote. Texts (titles, axes, legends) are Hungarian; code and comments are English.
"""

import re
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ------------------------------------------------------------------------------ labels
POLICY_HU = {
    "uncontrolled": "Szabályozatlan",
    "pv_following": "PV-követő szabály",
    "line_following": "Vezetékterhelés-követő szabály",
    "grid_guard": "Hálózatőr szabály (akkumulátorral)",
    "least_laxity": "Legkisebb tartalék-idő szabály",
    "oracle": "Elméleti optimum (előrelátó)",
    "ppo_pv_max": "RL – PV-maximalizáló",
    "ppo_grid_safe": "RL – hálózatbarát",
    "ppo_balanced": "RL – kiegyensúlyozott",
}
PROFILE_HU = {
    "pv_max": "PV-maximalizáló",
    "grid_safe": "hálózatbarát",
    "balanced": "kiegyensúlyozott",
}
COLORS = {
    "uncontrolled": "#8d8d86",
    "pv_following": "#2f9e7a",
    "line_following": "#c9a227",
    "grid_guard": "#8c564b",
    "least_laxity": "#6b8e9f",
    "oracle": "#1d1d1b",
    "ppo_pv_max": "#2f6fd0",
    "ppo_grid_safe": "#e07b1a",
    "ppo_balanced": "#7a4fc2",
}
INK, GRID = "#2b2b2b", "#dcdcd6"
SEED = re.compile(r"_s\d+$")
WIDE = (7.2, 3.3)
# label offsets (points) of the trade-off scatter, where markers lie close together
LABEL_OFFSET = {
    "pv_following": (6, -11),
    "ppo_pv_max": (-6, 7),
    "line_following": (7, 7),
    "ppo_balanced": (9, -4),
    "ppo_grid_safe": (-8, -4),
    "oracle": (-7, 8),
}

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.edgecolor": "#9a9a92",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}


def base(name: str) -> str:
    """Policy name without the seed suffix (``ppo_x_s2`` -> ``ppo_x``)."""
    return SEED.sub("", name)


def label(name: str) -> str:
    return POLICY_HU.get(base(name), base(name))


def _save(fig, out: Path, stem: str) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = (out / f"{stem}.{ext}").resolve()
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _panel_letters(axes) -> None:
    """(a), (b) ... in front of the panel titles (left aligned); a corner label if untitled."""
    for ax, ch in zip(np.ravel(axes), "abcdefgh"):
        title = ax.get_title() or ax.get_title(loc="left")
        if title:
            ax.set_title("", loc="center")  # the centred title would be drawn on top
            ax.set_title(f"({ch}) {title}", loc="left")
        else:
            ax.text(-0.02, 1.04, f"({ch})", transform=ax.transAxes, fontweight="bold", fontsize=9)


def seed_stats(kpis: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean over days per (policy, seed), then mean and std over the seeds of a profile."""
    per = kpis.groupby("policy", sort=False)[cols].mean()
    per["pol"] = [base(p) for p in per.index]
    g = per.groupby("pol", sort=False)[cols]
    return pd.concat({"mean": g.mean(), "std": g.std().fillna(0.0)}, axis=1)


# ------------------------------------------------------------------------------ 0 network
def fig_network(data, out: Path) -> list[Path]:
    """Single-line schematic of the campus: 10 kV bus, 5 transformers, feeders and the assets.

    ``data`` is the ``CampusData`` of the case study. PV roofs (orange squares), chargers
    (green triangles with their number), the battery (blue diamond) and the feeders that are
    overloaded by the PV export (red ring) are marked; every feeder carries its building.
    """
    from app.case_studies.kfki_campus.plots_network import build_layout

    lay = build_layout(data)
    label_hu = {r.bus: r.label_hu.replace(" (label unclear)", "") for r in data.loads.itertuples()}
    label_hu.update({b: "NaS akkumulátor" for b in data.battery["bus"]})
    n_ev = data.ev_sites["bus"].value_counts().to_dict()
    pv_roof = set(data.pv_sites["bus"])
    pv_small = set(data.pv["bus"])
    batt = set(data.battery["bus"])
    stressed = {"4T_b4", "4T_b26", "6T_b6"}  # overloaded by the PV export (figure 3)
    y_mv, y_tr, y_bar, y_node = 0.0, -1.1, -2.0, -3.0
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.4, 4.3))
        ax.axis("off")
        ax.set_xlim(lay.mv[0] - 3.2, max(lay.x[t] for t in lay.terminals) + 1.0)
        ax.set_ylim(-7.2, 1.6)
        # supply and the 10 kV bus
        ax.plot(lay.mv, [y_mv, y_mv], color=INK, lw=3)
        ax.annotate("", xy=(lay.mv[0], y_mv), xytext=(lay.mv[0] - 2.5, y_mv), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
        ax.text(lay.mv[0] - 2.6, y_mv + 0.25, "Virányos 10 kV-os\nbetáplálás (10 km)", ha="left", va="bottom", fontsize=7)
        ax.text(lay.x[lay.mv_bus], y_mv + 0.25, "10 kV-os gyűjtősín", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for t in data.transformers.itertuples():
            lv = lay.trafo_lv[t.id]
            tx = lay.trafo_x[t.id]
            name = t.id.replace("TR_", "")
            hot = t.id == "TR_4T"
            col = "#8c564b" if hot else INK
            ax.plot([tx, tx], [y_mv, y_tr + 0.3], color=INK, lw=1.2)
            for dx in (-0.12, 0.12):
                ax.add_patch(plt.Circle((tx + dx, y_tr), 0.2, fill=False, ec=col, lw=1.4))
            ax.plot([tx, tx], [y_tr - 0.2, y_bar], color=INK, lw=1.2)
            if hot:
                note = f"{name} trafó, {t.sn_kVA / 1000:g} MVA\nkorlát: 250 kVA"
            else:
                note = f"{name} trafó\n{t.sn_kVA / 1000:g} MVA"
            ax.text(tx + 0.55, y_tr + 0.15, note, ha="left", va="center", fontsize=6.2 if hot else 6.6, color=col, fontweight="bold" if hot else "normal")
            x0, x1 = lay.bars[lv]
            ax.plot([x0, x1], [y_bar, y_bar], color=INK, lw=2.4)
        for term in lay.terminals:
            x = lay.x[term]
            ax.plot([x, x], [y_bar, y_node], color=INK, lw=0.9)
            ax.plot(x, y_node, "o", ms=4.5, color="#555", zorder=3)
            if term in stressed:
                ax.plot(x, y_node, "o", ms=12, mfc="none", mec="#b22222", mew=1.6, zorder=2)
            y = y_node - 0.55
            if term in pv_roof or term in pv_small:
                ax.plot(x, y, "s", ms=6.5 if term in pv_roof else 4, color="#f2a900", mec=INK, mew=0.5, zorder=3)
                y -= 0.5
            if term in n_ev:
                ax.plot(x, y, "^", ms=8, color="#2f9e7a", mec=INK, mew=0.5, zorder=3)
                ax.text(x, y - 0.02, str(n_ev[term]), ha="center", va="center", fontsize=5.4, color="white", fontweight="bold", zorder=4)
                y -= 0.5
            if term in batt:
                ax.plot(x, y, "D", ms=6.5, color="#2f6fd0", mec=INK, mew=0.5, zorder=3)
                y -= 0.5
            ax.text(x, y - 0.1, label_hu.get(term, term), rotation=90, ha="center", va="top", fontsize=5.8)
        handles = [
            plt.Line2D([], [], marker="s", ls="", ms=6.5, color="#f2a900", mec=INK, mew=0.5, label="PV-tető (400 kWp épületenként)"),
            plt.Line2D([], [], marker="s", ls="", ms=4, color="#f2a900", mec=INK, mew=0.5, label="meglévő PV (9 kWp)"),
            plt.Line2D([], [], marker="^", ls="", ms=8, color="#2f9e7a", mec=INK, mew=0.5, label="elektromos töltők (22 kW, a szám a darabszám)"),
            plt.Line2D([], [], marker="D", ls="", ms=6.5, color="#2f6fd0", mec=INK, mew=0.5, label="NaS akkumulátor (1450 kWh, 250 kW)"),
            plt.Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec="#b22222", mew=1.6, label="a PV-betáplálás miatt túlterhelt vezeték"),
            plt.Line2D([], [], marker="o", ls="", ms=4.5, color="#555", label="fogyasztói leágazás"),
        ]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=7, columnspacing=1.5, handletextpad=0.4)
        ax.set_title("A KFKI campus középfeszültségű és kisfeszültségű hálózata", fontsize=9.5, fontweight="bold", loc="left")
        return _save(fig, out, "00_halozat_felepitese")


# ------------------------------------------------------------------------------ 1-2 GMM
def fig_gmm_sessions(sessions: pd.DataFrame, info: dict, out: Path) -> list[Path]:
    """Start hour, energy and stay of the sampled sessions, working days vs weekends."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
        spec = [
            ("start_hour", "Érkezés időpontja (óra)", np.arange(0, 25, 1.0)),
            ("energy_kwh", "Igényelt energia (kWh)", np.arange(0, 61, 2.5)),
            ("stay_h", "Tartózkodási idő (óra)", np.arange(0, 25, 1.0)),
        ]
        for ax, (col, xl, bins) in zip(axes, spec):
            for kind, color in (("workday", "#2f6fd0"), ("weekend", "#e07b1a")):
                d = sessions.loc[sessions["day_type"] == kind, col].clip(upper=bins[-1] - 1e-6)
                name = "munkanap" if kind == "workday" else "hétvége"
                ax.hist(d, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=color, label=name)
                ax.hist(d, bins=bins, density=True, histtype="step", color=color, lw=1.3)
            ax.set_xlabel(xl)
            ax.set_ylabel("relatív gyakoriság")
        axes[0].legend(loc="upper right")
        _panel_letters(axes)
        fig.suptitle(
            "A GMM-ekből mintavételezett töltési események "
            f"(illesztési minta: {info['n_fit_workday']} munkanapi, {info['n_fit_weekend']} hétvégi esemény)",
            fontsize=9,
            y=1.03,
        )
        fig.tight_layout()
        return _save(fig, out, "01_gmm_esemenyek")


def fig_gmm_load(curves: dict, out: Path) -> list[Path]:
    """Aggregate uncontrolled EV load of the 9 chargers: median, bands and sample days."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        hours = np.arange(24) + 0.5
        for ax, (kind, title) in zip(axes, (("workday", "Munkanap"), ("weekend", "Hétvége"))):
            m = curves[kind]  # days x 24 (kW)
            ax.fill_between(hours, np.percentile(m, 5, axis=0), np.percentile(m, 95, axis=0), color="#2f6fd0", alpha=0.15, lw=0, label="5–95 %-os sáv")
            ax.fill_between(hours, np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0), color="#2f6fd0", alpha=0.30, lw=0, label="25–75 %-os sáv")
            for row in m[:4]:
                ax.step(hours, row, where="mid", color="#8d8d86", lw=0.6, alpha=0.8)
            ax.plot(hours, np.median(m, axis=0), color="#1d1d1b", lw=1.6, label="medián")
            ax.plot([], [], color="#8d8d86", lw=0.8, label="példa napok")
            ax.set_title(title)
            ax.set_xlabel("Napszak (óra)")
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 4))
        axes[0].set_ylabel("Összes töltési teljesítmény (kW)")
        axes[0].legend(loc="upper right")
        _panel_letters(axes)
        fig.suptitle("Szabályozatlan töltés sztochasztikus terhelési profilja (9 töltő)", fontsize=9, y=1.02)
        fig.tight_layout()
        return _save(fig, out, "02_toltesi_terheles")


# ------------------------------------------------------------------------------ 3 campus
def fig_campus(hard_eps: dict, line_names: list[str], limit_kva: float, out: Path, trafo_idx: int = 2) -> list[Path]:
    """Where the campus is stressed: overload frequency and the worst day (uncontrolled)."""
    eps = {d: e for (p, d), e in hard_eps.items() if p == "uncontrolled"}
    ev_lines = list(next(iter(eps.values()))["ev_lines"])
    names = [line_names[i] for i in ev_lines]
    over = np.array(
        [[(np.nanmax(e["line_loading"][:, i]) > 100.0) for i in ev_lines] for e in eps.values()]
    ).mean(axis=0)
    tr = np.array([np.nanmax(e["trafo_loading"][:, trafo_idx]) * 1000.0 / limit_kva for e in eps.values()])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})
        ax = axes[0]
        labels = [n.replace("L_", "").replace("_", " ") + " vezeték" for n in names] + ["4T transzformátor\n(250 kVA korlát)"]
        vals = list(over * 100) + [float((tr > 100).mean() * 100)]
        cols = ["#c9a227"] * len(names) + ["#8c564b"]
        ax.barh(range(len(vals)), vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(v + 1.5, i, f"{v:.0f} %", va="center", fontsize=8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title("Túlterhelések gyakorisága")
        ax.set_xlabel("A vizsgált napok hányada (%)")
        ax.grid(axis="y", visible=False)
        # worst day for the transformer
        day = list(eps)[int(np.argmax(tr))]
        e = eps[day]
        hrs = np.arange(len(e["trafo_loading"])) + 0.5
        ax = axes[1]
        ax.plot(hrs, e["trafo_loading"][:, trafo_idx] * 1000.0 / limit_kva, color="#8c564b", lw=1.6, label="4T transzformátor (a korlát %-ában)")
        li = ev_lines[int(np.argmax([np.nanmax(e["line_loading"][:, i]) for i in ev_lines]))]
        ax.plot(hrs, e["line_loading"][:, li], color="#c9a227", lw=1.6, label=f"{line_names[li].replace('L_', '').replace('_', ' ')} vezeték (a névleges %-ában)")
        ax.axhline(100, color="#b22222", lw=1.0, ls="--")
        ax.text(0.4, 106, "100 % (határ)", color="#b22222", fontsize=7.5, ha="left")
        ax2 = ax.twinx()
        ax2.fill_between(hrs, 0, e["dispatch"]["pv"].to_numpy() / 1000.0, color="#f2c14e", alpha=0.25, lw=0)
        ax2.set_ylabel("PV-termelés (MW)", color="#9a7b16")
        ax2.set_ylim(0, 4)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlabel("Napszak (óra)")
        ax.set_ylabel("Terhelés (%)")
        ax.set_title("A legterheltebb nap")
        ax.set_ylim(-5, 195)
        ax.legend(loc="upper left", fontsize=7)
        _panel_letters(axes)
        fig.tight_layout()
        return _save(fig, out, "03_campus_kihivasok")


# ------------------------------------------------------------------------------ 4 standard
def _gain_pct(kpis: pd.DataFrame) -> pd.DataFrame:
    """Gain of every policy under every profile as % of the oracle's gain (per seed mean)."""
    g = [f"gain_{p}" for p in PROFILE_HU]
    per = kpis.groupby("policy", sort=False)[g].mean()
    orc = per.loc["oracle"]
    return per / orc * 100.0


def fig_rl_profiles(kpis: pd.DataFrame, out: Path) -> list[Path]:
    """Trade-off PV-use / feeder overload and the share of the oracle's gain per profile."""
    cols = ["ev_pv_share", "ev_line_overload_kvah"]
    st = seed_stats(kpis, cols)
    pct = _gain_pct(kpis)
    pct["pol"] = [base(p) for p in pct.index]
    pct_m = pct.groupby("pol", sort=False).agg(["mean", "std"]).fillna(0.0)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
        ax = axes[0]
        for pol in st.index:
            if pol in ("least_laxity", "random"):
                continue
            x, y = st.loc[pol, ("mean", cols[0])], st.loc[pol, ("mean", cols[1])]
            ax.errorbar(x, y, xerr=st.loc[pol, ("std", cols[0])], yerr=st.loc[pol, ("std", cols[1])], fmt="o" if pol.startswith("ppo") else "s", ms=6.5, color=COLORS.get(pol, "#555"), mec=INK, mew=0.5, capsize=2, lw=1)
            ax.plot([], [], "o" if pol.startswith("ppo") else "s", ms=6, color=COLORS.get(pol, "#555"), mec=INK, mew=0.5, label=POLICY_HU.get(pol, pol))
        lo, hi = st[("mean", cols[0])].min(), st[("mean", cols[0])].max()
        ax.set_xlim(lo - 0.035, hi + 0.04)
        ax.set_xlabel("Töltött energia PV-többletben (hányad)")
        ax.set_ylabel("Töltő-vezetékek túlterhelése (kVAh/nap)")
        ax.set_title("Célok közti kompromisszum")
        ax.text(0.97, 0.06, "jobb ↘", transform=ax.transAxes, ha="right", fontsize=8, color="#555")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.8, columnspacing=1.0, handletextpad=0.3)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax = axes[1]
        profiles = list(PROFILE_HU)
        groups = [("pv_following", "PV-követő szabály"), ("line_following", "Vezetékterhelés-\nkövető szabály"), ("RL", "RL (a profilhoz\ntanítva)")]
        w = 0.26
        for gi, (pol, name) in enumerate(groups):
            vals, errs, cols_ = [], [], []
            for p in profiles:
                key = f"ppo_{p}" if pol == "RL" else pol
                vals.append(pct_m.loc[key, (f"gain_{p}", "mean")])
                errs.append(pct_m.loc[key, (f"gain_{p}", "std")])
                cols_.append(COLORS[f"ppo_{p}"] if pol == "RL" else COLORS[pol])
            ax.bar(np.arange(3) + (gi - 1) * w, vals, w * 0.92, yerr=errs, color=cols_, capsize=2, edgecolor=INK, linewidth=0.4, alpha=0.92 if pol == "RL" else 0.75, hatch="" if pol == "RL" else "//")
        ax.axhline(100, color=COLORS["oracle"], lw=1.2, ls="--")
        ax.text(2.45, 102, "elméleti optimum = 100 %", ha="right", va="bottom", fontsize=7.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{PROFILE_HU[p]}\ncélfüggvény" for p in profiles], fontsize=6.8)
        ax.set_ylabel("Az optimum előnyének hányada (%)")
        ax.set_ylim(0, 112)
        ax.set_title("Az elérhető előny kihasználása")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS["pv_following"], hatch="//", alpha=0.75), plt.Rectangle((0, 0), 1, 1, fc=COLORS["line_following"], hatch="//", alpha=0.75), plt.Rectangle((0, 0), 1, 1, fc="#7a4fc2")]
        ax.legend(handles, [g[1].replace("\n", "") for g in groups], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=6.8)
        _panel_letters(axes)
        fig.tight_layout()
        return _save(fig, out, "04_rl_profilok")


# ------------------------------------------------------------------------------ 5 example day
def pick_overload_day(episodes: dict, policy: str = "uncontrolled") -> int:
    """Day with the highest feeder overload of the uncontrolled run."""
    best, best_day = -1.0, None
    for (p, d), e in episodes.items():
        if p != policy:
            continue
        v = float(np.nansum(np.maximum(e["line_loading"][:, e["ev_lines"]] - 100.0, 0.0)))
        if v > best:
            best, best_day = v, d
    return best_day


def pick_example_day(episodes: dict, other: str = "oracle") -> tuple[int, int]:
    """(day, line) where the controlled run relieves a charger feeder most (kVAh above 100 %)."""
    best, pick = -1.0, (None, None)
    for (p, d), e in episodes.items():
        if p != "uncontrolled" or (other, d) not in episodes:
            continue
        o = episodes[(other, d)]
        for i in e["ev_lines"]:
            over = lambda x: float(np.nansum(np.maximum(x["line_loading"][:, i] - 100.0, 0.0)))
            gain = over(e) - over(o)
            if gain > best:
                best, pick = gain, (d, int(i))
    return pick


def fig_example_day(episodes: dict, line_names: list[str], out: Path, day: Optional[int] = None) -> list[Path]:
    """EV power and the loading of a charger feeder on the day where control helps most."""
    day, li = pick_example_day(episodes) if day is None else (day, None)
    pols = [p for p in ("uncontrolled", "ppo_pv_max_s0", "ppo_grid_safe_s0", "oracle") if (p, day) in episodes]
    unc = episodes[("uncontrolled", day)]
    if li is None:
        li = int(unc["ev_lines"][int(np.argmax([np.nanmax(unc["line_loading"][:, i]) for i in unc["ev_lines"]]))])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.1, 1.1]})
        hrs = np.arange(24) + 0.5
        pv = unc["dispatch"]["pv"].to_numpy() / 1000.0
        axes[0].fill_between(hrs, 0, pv, color="#f2c14e", alpha=0.45, lw=0)
        axes[0].plot(hrs, unc["dispatch"]["demand"].to_numpy() / 1000.0 - unc["ev_power"].sum(axis=1).to_numpy() / 1000.0, color="#555", lw=1.2)
        axes[0].set_ylabel("Teljesítmény (MW)")
        axes[0].set_title("PV-termelés (sárga) és az épületek fogyasztása (szürke)")
        for pol in pols:
            e = episodes[(pol, day)]
            kw = dict(color=COLORS[base(pol)], lw=1.7 if base(pol) != "uncontrolled" else 1.4, ls="--" if pol == "oracle" else "-", label=label(pol))
            axes[1].step(hrs, e["ev_power"].sum(axis=1).to_numpy(), where="mid", **kw)
            axes[2].plot(hrs, e["line_loading"][:, li], **kw)
        axes[1].set_ylabel("Töltési teljesítmény (kW)")
        axes[1].set_title("Az összes töltő teljesítménye")
        h, lab = axes[1].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.025))
        axes[2].axhline(100, color="#b22222", lw=1.0, ls="--")
        axes[2].set_ylabel("Vezetékterhelés (%)")
        axes[2].set_title(f"A {line_names[li].replace('L_', '').replace('_', ' ')} vezeték terhelése (100 % = névleges)")
        axes[2].set_xlabel("Napszak (óra)")
        axes[2].set_xlim(4, 20)
        axes[2].set_xticks(range(4, 21, 2))
        axes[2].set_ylim(50, 125)
        _panel_letters(axes)
        fig.tight_layout()
        return _save(fig, out, "05_pelda_nap")


# ------------------------------------------------------------------------------ 6 hard
def fig_battery_trafo(kpis: pd.DataFrame, episodes: dict, limit_kva: float, out: Path, day: Optional[int] = None, trafo_idx: int = 2) -> list[Path]:
    """Battery + transformer limit: overload by policy and one day of the 4T transformer."""
    cols = ["trafo_overload_kvah", "import_kwh"]
    st = seed_stats(kpis, cols)
    order = ["uncontrolled", "line_following", "grid_guard", "ppo_pv_max", "ppo_balanced", "ppo_grid_safe", "oracle"]
    order = [p for p in order if p in st.index]
    if day is None:  # day with the largest uncontrolled transformer overload
        tot = {d: float(np.nansum(np.maximum(e["trafo_loading"][:, trafo_idx] * 1000.0 / limit_kva - 100.0, 0.0))) for (p, d), e in episodes.items() if p == "uncontrolled"}
        day = max(tot, key=tot.get)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.4, 4.9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1])
        a0 = fig.add_subplot(gs[:, 0])
        vals = [st.loc[p, ("mean", cols[0])] for p in order]
        errs = [st.loc[p, ("std", cols[0])] for p in order]
        a0.barh(range(len(order)), vals, xerr=errs, color=[COLORS[p] for p in order], edgecolor=INK, linewidth=0.4, capsize=2)
        for i, (v, er) in enumerate(zip(vals, errs)):
            a0.text(v + er + 2, i, f"{v:.0f}", va="center", fontsize=7.5)
        a0.set_xlim(0, max(vals) * 1.15)
        a0.set_yticks(range(len(order)))
        a0.set_yticklabels([POLICY_HU[p] for p in order], fontsize=7.2)
        a0.invert_yaxis()
        a0.set_xlabel("4T transzformátor túlterhelése (kVAh/nap)")
        a0.set_title("Átlagos napi túlterhelés")
        a0.grid(axis="y", visible=False)
        a1 = fig.add_subplot(gs[0, 1])
        a2 = fig.add_subplot(gs[1, 1], sharex=a1)
        hrs = np.arange(24) + 0.5
        for pol in ("uncontrolled", "grid_guard", "ppo_grid_safe_s0", "oracle"):
            if (pol, day) not in episodes:
                continue
            e = episodes[(pol, day)]
            kw = dict(color=COLORS[base(pol)], lw=1.6, ls="--" if pol == "oracle" else "-", label=label(pol))
            a1.plot(hrs, e["trafo_loading"][:, trafo_idx] * 1000.0 / limit_kva, **kw)
            d = e["dispatch"]
            a2.step(hrs, (d["p_charge"] - d["p_discharge"]).to_numpy(), where="mid", **kw)
        a1.axhline(100, color="#b22222", lw=1.0, ls="--")
        a1.set_ylabel("4T terhelés (a korlát %-a)")
        a1.set_title("A legterheltebb nap")
        a1.set_ylim(0, 175)
        h, lab = a1.get_legend_handles_labels()
        a2.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=6.6, columnspacing=1.0, handlelength=1.6)
        a2.axhline(0, color="#777", lw=0.6)
        a2.set_ylabel("Akkumulátor (kW)\n+ töltés / − kisütés")
        a2.set_xlabel("Napszak (óra)")
        a2.set_xlim(0, 24)
        a2.set_xticks(range(0, 25, 4))
        a2.set_title("Az akkumulátor teljesítménye")
        _panel_letters([a0, a1, a2])
        fig.tight_layout()
        return _save(fig, out, "06_akkumulator_trafo")


# ------------------------------------------------------------------------------ 7 sensitivity
def fig_sensitivity(sens: pd.DataFrame, out: Path) -> list[Path]:
    """Benefit against the number of sessions, the flexibility (stay) and the PV size."""
    axes_def = [
        ("n", "Kiszolgált töltések (db/nap)", "sessions"),
        ("flex", "Tartózkodási idő szorzója", "value"),
        ("pv", "Hozzáadott PV (kWp)", "value"),
    ]
    pols = [("line_following", "Vezetékterhelés-követő szabály", "s"), ("ppo_grid_safe", "RL – hálózatbarát", "o"), ("oracle", "Elméleti optimum", "D")]
    sens = sens.copy()
    sens["pol"] = sens["policy"].map(base)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0), sharey="row")
        for j, (axis, xl, xcol) in enumerate(axes_def):
            d = sens[sens["axis"] == axis]
            for i, (metric, yl) in enumerate((("ev_line_overload_kvah", "Vezetéktúlterhelés\ncsökkenése (%)"), ("ev_pv_share", "PV-ben töltött hányad\nváltozása (százalékpont)"))):
                ax = axes[i][j]
                unc = d[d["pol"] == "uncontrolled"].groupby("value")[metric].mean()
                xs_map = d[d["pol"] == "uncontrolled"].groupby("value")["ev_sessions"].mean()
                for pol, name, mk in pols:
                    # per value and seed: mean over days; then mean and spread over seeds
                    per = d[d["pol"] == pol].groupby(["value", "policy"])[metric].mean().reset_index()
                    if metric == "ev_line_overload_kvah":
                        per["y"] = (1 - per[metric] / per["value"].map(unc)) * 100
                    else:
                        per["y"] = (per[metric] - per["value"].map(unc)) * 100
                    agg = per.groupby("value")["y"].agg(["mean", "min", "max"])
                    xs = xs_map.loc[agg.index] if xcol == "sessions" else agg.index.to_numpy()
                    ax.plot(xs, agg["mean"], marker=mk, ms=5, color=COLORS[pol], lw=1.4, label=name)
                    if pol.startswith("ppo") and (agg["max"] - agg["min"]).max() > 0:
                        ax.fill_between(xs, agg["min"], agg["max"], color=COLORS[pol], alpha=0.2, lw=0)
                ax.set_ylabel(yl if j == 0 else "")
                if i == 1:
                    ax.set_xlabel(xl)
                if axis == "pv":
                    ax.set_xticks([1000, 1500, 2000, 3000])
                ax.axhline(0, color="#777", lw=0.6)
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, -0.04))
        axes[0][0].set_title("Járművek száma")
        axes[0][1].set_title("Rugalmasság")
        axes[0][2].set_title("PV-kapacitás")
        fig.suptitle("Érzékenységvizsgálat a szabályozatlan töltéshez képest (hálózatbarát célfüggvény)", fontsize=9, y=1.01)
        _panel_letters(axes)
        fig.tight_layout()
        return _save(fig, out, "07_erzekenyseg")


# ------------------------------------------------------------------------------ table
def result_table(kpis: dict, out: Path) -> Path:
    """Key numbers of the standard and the hard environment as CSV and a LaTeX table."""
    rows = []
    for env, name in (("standard", "Alap eset"), ("hard", "Akkumulátor + 4T korlát")):
        k = kpis[env]
        cols = ["ev_pv_share", "ev_line_overload_kvah", "trafo_overload_kvah", "ev_unmet_kwh"]
        st = seed_stats(k, cols)
        pct = _gain_pct(k)
        pct["pol"] = [base(p) for p in pct.index]
        pm = pct.groupby("pol", sort=False).mean()
        for pol in st.index:
            if pol in ("random", "least_laxity"):
                continue
            prof = pol.replace("ppo_", "") if pol.startswith("ppo_") else None
            own = pm.loc[pol, f"gain_{prof}"] if prof else np.nan
            rows.append({"Eset": name, "Stratégia": POLICY_HU.get(pol, pol), "PV-ben töltött hányad": st.loc[pol, ("mean", cols[0])], "Vezetéktúlterhelés (kVAh/nap)": st.loc[pol, ("mean", cols[1])], "Trafótúlterhelés (kVAh/nap)": st.loc[pol, ("mean", cols[2])], "Ki nem szolgált energia (kWh/nap)": st.loc[pol, ("mean", cols[3])], "Előny az optimum %-ában (saját profil)": own})
    df = pd.DataFrame(rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out / "tablazat.csv", index=False, encoding="utf-8-sig")
    with open(out / "tablazat.tex", "w", encoding="utf-8") as f:
        f.write(df.round(2).to_latex(index=False, na_rep="–", float_format="%.2f", column_format="llrrrrr"))
    return out / "tablazat.csv"
