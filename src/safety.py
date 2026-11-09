"""Part 2: Tesla autopilot deaths. Load the messy CSV, clean it, answer the problem statement.

Cleaning decisions (all visible on the /safety page):
- column names carry leading/trailing spaces; stripped.
- two unnamed columns and the Source, Note and Deceased 1-4 columns are dropped (not analytic).
- 13 footer rows have no Case #: yearly totals, a "2013..2022" header row and a note
  about the AutoPilot release date. Dropped; totals are recomputed from the events.
- '-' in count columns means 0.
- one event has Year=202 (typo); Year is re-derived from Date for every row.
- 'Holland' is folded into 'Netherlands'.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = (
    Path(os.environ.get("DATA_ROOT", Path.home() / "Caltech CTME" / "PGC AIML - Capstone" / "Datasets"))
    / "Capstone 1"
    / "Part 2"
    / "Tesla - Deaths.csv"
)
STATIC_DIR = PROJECT_DIR / "static"

COUNT_COLS = [
    "Deaths",
    "Tesla driver",
    "Tesla occupant",
    "Other vehicle",
    "Cyclists/ Peds",
    "TSLA+cycl / peds",
    "Autopilot claimed",
    "Verified Tesla Autopilot Deaths",
    "Verified Tesla Autopilot Deaths + All Deaths Reported to NHTSA SGO",
]
SHORT = {
    "Verified Tesla Autopilot Deaths": "Verified autopilot deaths",
    "Verified Tesla Autopilot Deaths + All Deaths Reported to NHTSA SGO": "Verified + NHTSA SGO",
    "TSLA+cycl / peds": "Tesla + cyclist/ped",
}

# chart colours on the shared dark panel (common/ui.css)
PANEL = "#182033"
INK = "#e8ecf6"
INK_2 = "#7c869d"
ACCENT = "#4fd1c5"
WARN = "#f2b84b"


def load_raw(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    inspection = {
        "raw_rows": len(raw),
        "raw_columns": int(raw.shape[1]),
        "dropped_columns": [c for c in df.columns if c.startswith("Unnamed")]
        + ["Source", "Note", "Deceased 1", "Deceased 2", "Deceased 3", "Deceased 4"],
    }
    df = df.drop(columns=[c for c in inspection["dropped_columns"] if c in df.columns])
    footer = df["Case #"].isna()
    inspection["footer_rows_dropped"] = int(footer.sum())
    df = df[~footer].copy()
    inspection["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    for c in ["Country", "State", "Model", "Description"]:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "-", "": "-"})
    df["Country"] = df["Country"].replace({"Holland": "Netherlands"})
    for c in COUNT_COLS:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip().replace("-", "0"), errors="coerce").fillna(0).astype(int)
    df["Case #"] = df["Case #"].astype(int)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["Year"] = df["Date"].dt.year.astype("Int64")
    df["Model"] = df["Model"].where(df["Model"].isin(["S", "3", "X", "Y"]), "unknown")
    inspection["events"] = len(df)
    inspection["missing_dates"] = int(df["Date"].isna().sum())
    inspection["date_range"] = [str(df["Date"].min().date()), str(df["Date"].max().date())]
    return df.sort_values("Date").reset_index(drop=True), inspection


def analyze(df: pd.DataFrame) -> dict:
    by_year = (
        df.groupby("Year")
        .agg(
            events=("Case #", "count"),
            deaths=("Deaths", "sum"),
            autopilot_claimed=("Autopilot claimed", "sum"),
            verified_autopilot=("Verified Tesla Autopilot Deaths", "sum"),
            verified_plus_nhtsa=("Verified Tesla Autopilot Deaths + All Deaths Reported to NHTSA SGO", "sum"),
        )
        .reset_index()
    )
    by_country = (
        df.groupby("Country")
        .agg(events=("Case #", "count"), deaths=("Deaths", "sum"))
        .sort_values("events", ascending=False)
        .reset_index()
    )
    usa = df[(df["Country"] == "USA") & (df["State"] != "-")]
    by_state = (
        usa.groupby("State")
        .agg(events=("Case #", "count"), deaths=("Deaths", "sum"))
        .sort_values("events", ascending=False)
        .reset_index()
    )
    by_model = (
        df.groupby("Model")
        .agg(
            events=("Case #", "count"),
            deaths=("Deaths", "sum"),
            verified_autopilot=("Verified Tesla Autopilot Deaths", "sum"),
        )
        .sort_values("events", ascending=False)
        .reset_index()
    )
    by_weekday = df.groupby(df["Date"].dt.day_name()).size()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_weekday = by_weekday.reindex(weekday_order).fillna(0).astype(int).reset_index()
    by_weekday.columns = ["weekday", "events"]
    victims = df["Deaths"].value_counts().sort_index().reset_index()
    victims.columns = ["deaths_in_event", "events"]

    n = len(df)
    occupant_any = (df["Tesla driver"] + df["Tesla occupant"]) > 0
    cyc_ped = df["Cyclists/ Peds"] > 0
    other_vehicle = df["Other vehicle"] > 0
    verified_total = df["Verified Tesla Autopilot Deaths"].sum()
    summary = {
        "events": int(n),
        "deaths": int(df["Deaths"].sum()),
        "tesla_driver_deaths": int(df["Tesla driver"].sum()),
        "events_tesla_driver_died": int((df["Tesla driver"] > 0).sum()),
        "tesla_occupant_deaths": int(df["Tesla occupant"].sum()),
        "events_with_occupant_death": int(occupant_any.sum()),
        "pct_events_with_occupant_death": round(100 * occupant_any.mean(), 1),
        "events_cyclist_or_pedestrian": int(cyc_ped.sum()),
        "pct_events_cyclist_or_pedestrian": round(100 * cyc_ped.mean(), 1),
        "cyclist_pedestrian_deaths": int(df["Cyclists/ Peds"].sum()),
        "events_occupant_and_cyc_ped": int((occupant_any & cyc_ped).sum()),
        "events_other_vehicle": int(other_vehicle.sum()),
        "pct_events_other_vehicle": round(100 * other_vehicle.mean(), 1),
        "other_vehicle_deaths": int(df["Other vehicle"].sum()),
        "autopilot_claimed_deaths": int(df["Autopilot claimed"].sum()),
        "events_autopilot_claimed": int((df["Autopilot claimed"] > 0).sum()),
        "verified_autopilot_deaths": int(df["Verified Tesla Autopilot Deaths"].sum()),
        "events_verified_autopilot": int((df["Verified Tesla Autopilot Deaths"] > 0).sum()),
        "verified_plus_nhtsa_deaths": int(df[COUNT_COLS[-1]].sum()),
        "pct_deaths_verified_autopilot": round(100 * verified_total / max(df["Deaths"].sum(), 1), 1),
        "single_death_events": int((df["Deaths"] == 1).sum()),
        "pct_single_death_events": round(100 * (df["Deaths"] == 1).mean(), 1),
        "usa_share_pct": round(100 * (df["Country"] == "USA").mean(), 1),
        "model_known_pct": round(100 * (df["Model"] != "unknown").mean(), 1),
        "peak_year": int(by_year.loc[by_year["events"].idxmax(), "Year"]),
        "peak_year_events": int(by_year["events"].max()),
    }
    return {
        "summary": summary,
        "by_year": by_year,
        "by_country": by_country,
        "by_state": by_state,
        "by_model": by_model,
        "by_weekday": by_weekday,
        "victims": victims,
    }


def _style(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(PANEL)
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=10)
    ax.tick_params(colors=INK_2, labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.yaxis.grid(True, color="#2a3347", linewidth=0.8)
    ax.set_axisbelow(True)


def render_charts(analysis: dict, out_dir: Path = STATIC_DIR) -> dict[str, str]:
    out_dir.mkdir(exist_ok=True)
    files = {}

    y = analysis["by_year"].dropna(subset=["Year"])
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    ax.bar(y["Year"].astype(int), y["deaths"], color=ACCENT, width=0.7)
    for yr, d in zip(y["Year"].astype(int), y["deaths"], strict=True):
        ax.text(yr, d + 1, str(d), ha="center", color=INK, fontsize=8)
    _style(ax, "Deaths in Tesla crashes per year")
    ax.set_xticks(y["Year"].astype(int))
    fig.tight_layout()
    fig.savefig(out_dir / "safety_deaths_per_year.png")
    plt.close(fig)
    files["deaths_per_year"] = "safety_deaths_per_year.png"

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    yrs = y["Year"].astype(int)
    ax.bar(yrs - 0.2, y["autopilot_claimed"], width=0.38, color=WARN, label="Autopilot claimed")
    ax.bar(yrs + 0.2, y["verified_autopilot"], width=0.38, color=ACCENT, label="Verified autopilot deaths")
    _style(ax, "Autopilot claimed vs verified, per year")
    ax.set_xticks(yrs)
    leg = ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    leg.get_frame().set_facecolor(PANEL)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_autopilot_claimed_vs_verified.png")
    plt.close(fig)
    files["autopilot"] = "safety_autopilot_claimed_vs_verified.png"

    m = analysis["by_model"]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=130)
    labels = ["Model " + v if v != "unknown" else "unknown" for v in m["Model"]]
    ax.barh(labels[::-1], m["events"][::-1], color=ACCENT, height=0.6)
    for i, v in enumerate(m["events"][::-1]):
        ax.text(v + 1, i, str(v), va="center", color=INK, fontsize=8)
    _style(ax, "Fatal events by Tesla model")
    ax.xaxis.grid(True, color="#2a3347", linewidth=0.8)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_by_model.png")
    plt.close(fig)
    files["by_model"] = "safety_by_model.png"
    return files


def narrative(s: dict) -> list[str]:
    return [
        (
            f"The sheet records {s['events']} fatal crashes involving a Tesla between 2013 and early 2023, "
            f"with {s['deaths']} deaths in total. {s['usa_share_pct']}% of events are in the USA, and California "
            "alone accounts for the largest share of US events, which tracks where Teslas are sold rather than "
            "any road effect."
        ),
        (
            f"Events rise every year with the fleet: {s['peak_year']} is the peak with {s['peak_year_events']} "
            "events. The data stops in January 2023, so 2023 is a partial year and should not be read as a drop."
        ),
        (
            f"Most crashes kill one person: {s['single_death_events']} of {s['events']} events "
            f"({s['pct_single_death_events']}%). The Tesla driver died in {s['events_tesla_driver_died']} events "
            f"and a Tesla occupant (driver or passenger) died in {s['pct_events_with_occupant_death']}% of events. "
            f"A cyclist or pedestrian was killed in {s['events_cyclist_or_pedestrian']} events "
            f"({s['pct_events_cyclist_or_pedestrian']}%), and in {s['events_occupant_and_cyc_ped']} of those a "
            f"Tesla occupant died as well. Another vehicle's occupant died in {s['events_other_vehicle']} events "
            f"({s['pct_events_other_vehicle']}%)."
        ),
        (
            f"Autopilot: it was claimed to be active in {s['events_autopilot_claimed']} events "
            f"({s['autopilot_claimed_deaths']} deaths) but verified in only {s['events_verified_autopilot']} "
            f"({s['verified_autopilot_deaths']} deaths, {s['pct_deaths_verified_autopilot']}% of all deaths). "
            f"Adding NHTSA Standing General Order reports brings the count to {s['verified_plus_nhtsa_deaths']} "
            "deaths. So verified Autopilot involvement is a small slice of Tesla fatalities, and the gap between "
            "claimed and verified is the main caveat of any safety conclusion drawn from this sheet."
        ),
        (
            f"Model is known for only {s['model_known_pct']}% of events; among known models, S and 3 dominate, "
            "which again follows sales volume. Nothing in this dataset normalises by miles driven or fleet size, "
            "so it cannot say whether Autopilot is safer or more dangerous than a human driver; it only counts "
            "events."
        ),
    ]


_cache: dict | None = None


def get_report(csv_path: Path = DEFAULT_CSV) -> dict:
    global _cache
    if _cache is None:
        df, inspection = clean(load_raw(csv_path))
        analysis = analyze(df)
        charts = render_charts(analysis)
        _cache = {
            "inspection": inspection,
            "summary": analysis["summary"],
            "tables": {k: v for k, v in analysis.items() if k != "summary"},
            "charts": charts,
            "narrative": narrative(analysis["summary"]),
        }
    return _cache
