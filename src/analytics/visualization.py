"""
src/analytics/visualization.py
------------------------------
Five publication-quality charts saved to assets/.

Public API (contract with src/run_live.py):
    generate_charts(master_df) -> List[Path]

Individual chart functions are exposed for notebook re-use:
    plot_correlation_heatmap(df, out_dir)
    plot_top_cities_dot(df, out_dir)
    plot_aqi_category_distribution(df, out_dir)
    plot_pollutant_dominance_stacked(df, out_dir)
    plot_aqi_time_series(df, out_dir)

All filenames are deterministic, so re-running overwrites cleanly.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; CI has no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Path bootstrap
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# Global style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.autolayout": False,
    "savefig.dpi": 150,
    "figure.dpi": 110,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
})

# CPCB category order + palette (matches AQI_Bucket values in cleaner)
AQI_CATEGORY_ORDER = [
    "Good", "Satisfactory", "Moderate", "Poor", "Very_Poor", "Severe"
]
AQI_CATEGORY_PALETTE = {
    "Good":         "#00b050",
    "Satisfactory": "#92d050",
    "Moderate":     "#ffc000",
    "Poor":         "#ff6600",
    "Very_Poor":    "#c00000",
    "Severe":       "#7c0000",
}

# Numeric pollutant columns the heatmap / dominance chart work with
POLLUTANT_NUMERIC_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_columns(df: pd.DataFrame, required: List[str]) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Skipping chart — missing columns: %s", missing)
        return False
    return True


def _save(fig: plt.Figure, out_path: Path, tight: bool = True) -> Path:
    fig.savefig(out_path, bbox_inches="tight" if tight else None)
    plt.close(fig)
    logger.info("Saved chart -> %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Chart 1: Masked Correlation Heatmap
# --------------------------------------------------------------------------- #
def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path = ASSETS_DIR) -> Optional[Path]:
    cols = [c for c in POLLUTANT_NUMERIC_COLS if c in df.columns]
    if len(cols) < 2:
        logger.warning("Correlation heatmap skipped — need ≥2 pollutant columns.")
        return None

    corr = df[cols].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # hide upper triangle

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 10},
        linewidths=0.5, linecolor="white",
        square=True, cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Pollutant Correlation Matrix (lower triangle)", pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return _save(fig, out_dir / "01_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Chart 2: Top 10 Polluted Cities — Dot plot
# --------------------------------------------------------------------------- #
def plot_top_cities_dot(df: pd.DataFrame, out_dir: Path = ASSETS_DIR) -> Optional[Path]:
    if not _ensure_columns(df, ["city", "Overall_AQI"]):
        return None

    city_stats = (
        df.dropna(subset=["Overall_AQI"])
          .groupby("city", as_index=False)["Overall_AQI"]
          .mean()
          .rename(columns={"Overall_AQI": "mean_aqi"})
          .sort_values("mean_aqi", ascending=False)
          .head(10)
    )
    if city_stats.empty:
        logger.warning("Top-cities dot plot skipped — no valid AQI values.")
        return None

    # Color each point by its AQI bucket (helps communicate severity)
    def _bucket(v: float) -> str:
        if v <= 50: return "Good"
        if v <= 100: return "Satisfactory"
        if v <= 200: return "Moderate"
        if v <= 300: return "Poor"
        if v <= 400: return "Very_Poor"
        return "Severe"
    city_stats["bucket"] = city_stats["mean_aqi"].apply(_bucket)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # Reverse so worst is at top in the visual y-axis
    plot_df = city_stats.iloc[::-1].reset_index(drop=True)

    sns.scatterplot(
        data=plot_df,
        x="mean_aqi", y="city",
        hue="bucket", size="mean_aqi",
        palette=AQI_CATEGORY_PALETTE,
        hue_order=AQI_CATEGORY_ORDER,
        sizes=(150, 450),
        edgecolor="white", linewidth=1.2,
        ax=ax, legend="brief",
    )
    ax.set_xlim(0, 500)
    ax.axvline(100, ls="--", lw=1.2, color="#92d050", alpha=0.85)
    ax.axvline(200, ls="--", lw=1.2, color="#ff6600", alpha=0.85)
    ax.text(101, -0.75, " Satisfactory (100)", color="#5a8a2a", fontsize=9)
    ax.text(201, -0.75, " Poor (200)", color="#cc5500", fontsize=9)

    ax.set_xlabel("Mean AQI (0–500)")
    ax.set_ylabel("")
    ax.set_title("Top 10 Most Polluted Cities (Mean AQI)", pad=14)
    ax.legend(title="AQI Category", bbox_to_anchor=(1.05, 1), loc="upper left",
              frameon=True, ncol=1)

    # Value labels
    for _, row in plot_df.iterrows():
        ax.text(row["mean_aqi"] + 5, row["city"], f"{row['mean_aqi']:.0f}",
                va="center", fontsize=9, color="#333333")

    return _save(fig, out_dir / "02_top_cities_dot_plot.png")


# --------------------------------------------------------------------------- #
# Chart 3: AQI Category Distribution
# --------------------------------------------------------------------------- #
def plot_aqi_category_distribution(df: pd.DataFrame, out_dir: Path = ASSETS_DIR) -> Optional[Path]:
    if not _ensure_columns(df, ["AQI_Bucket"]):
        return None

    counts = (
        df["AQI_Bucket"]
          .value_counts()
          .reindex(AQI_CATEGORY_ORDER, fill_value=0)
    )
    if counts.sum() == 0:
        logger.warning("AQI category distribution skipped — no bucket values.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=counts.index, y=counts.values,
        palette=[AQI_CATEGORY_PALETTE[c] for c in counts.index],
        edgecolor="black", linewidth=0.6,
        ax=ax,
    )
    ax.set_title("Station Distribution by AQI Category", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Stations")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(0.3, 0.015 * counts.max()), str(int(v)),
                ha="center", fontsize=10, fontweight="bold")
    sns.despine()
    return _save(fig, out_dir / "03_aqi_category_distribution.png")


# --------------------------------------------------------------------------- #
# Chart 4: Pollutant Dominance — Stacked horizontal bar (Top 5 cities)
# --------------------------------------------------------------------------- #
def plot_pollutant_dominance_stacked(df: pd.DataFrame,
                                     out_dir: Path = ASSETS_DIR) -> Optional[Path]:
    """
    Top-5 worst cities by mean AQI. Bar segments = contributions of each
    pollutant sub-index (mean). Segment boundary at 200 shows Poor threshold.
    """
    required = ["city", "Overall_AQI"] + [f"{p}_subindex" for p in POLLUTANT_NUMERIC_COLS]
    if not _ensure_columns(df, required):
        return None

    city_mean = (
        df.dropna(subset=["Overall_AQI"])
          .groupby("city", as_index=False)["Overall_AQI"].mean()
          .sort_values("Overall_AQI", ascending=False)
          .head(5)
    )
    if city_mean.empty:
        logger.warning("Pollutant dominance skipped — no valid AQI values.")
        return None

    top_cities = city_mean["city"].tolist()
    subcols = [f"{p}_subindex" for p in POLLUTANT_NUMERIC_COLS]
    sub_by_city = (
        df[df["city"].isin(top_cities)]
          .groupby("city")[subcols]
          .mean()
          .fillna(0.0)
          .loc[top_cities]
    )
    sub_by_city.columns = POLLUTANT_NUMERIC_COLS

    # Worst city at top: reverse row order
    plot_df = sub_by_city.iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    left = np.zeros(len(plot_df))
    colors = sns.color_palette("flare", n_colors=len(POLLUTANT_NUMERIC_COLS))

    for i, pol in enumerate(POLLUTANT_NUMERIC_COLS):
        vals = plot_df[pol].values
        ax.barh(plot_df.index, vals, left=left, label=pol,
                color=colors[i], edgecolor="white", linewidth=0.6)
        left = left + vals

    ax.axvline(200, ls="--", lw=1.4, color="#111111", alpha=0.7)
    ax.text(203, -0.6, " Poor threshold (200)", fontsize=9, color="#111111")

    ax.set_xlabel("Mean pollutant sub-index contribution")
    ax.set_ylabel("")
    ax.set_title("Pollutant Dominance — Top 5 Worst Cities", pad=14)
    ax.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc="upper left",
              frameon=True)
    sns.despine()
    return _save(fig, out_dir / "04_pollutant_dominance_stacked.png")


# --------------------------------------------------------------------------- #
# Chart 5: AQI Time Series (national hourly mean + 24h rolling mean)
# --------------------------------------------------------------------------- #
def plot_aqi_time_series(df: pd.DataFrame, out_dir: Path = ASSETS_DIR) -> Optional[Path]:
    if not _ensure_columns(df, ["timestamp_utc", "Overall_AQI"]):
        return None

    ts = df.copy()
    ts["timestamp_utc"] = pd.to_datetime(ts["timestamp_utc"], errors="coerce", utc=True)
    ts = ts.dropna(subset=["timestamp_utc", "Overall_AQI"])
    if ts["timestamp_utc"].nunique() <= 1:
        logger.info("Time-series skipped — dataset has ≤1 unique timestamp.")
        return None

    hourly = (
        ts.set_index("timestamp_utc")["Overall_AQI"]
          .resample("1h").mean()
          .dropna()
    )
    if hourly.empty:
        logger.warning("Time-series skipped — no hourly aggregation produced.")
        return None

    rolling = hourly.rolling(window=24, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly.index, hourly.values, color="#1f77b4", lw=1.2,
            alpha=0.75, label="Hourly mean AQI")
    ax.plot(rolling.index, rolling.values, color="#d62728", lw=2.2,
            label="24-hour rolling mean")
    ax.axhline(100, ls="--", lw=1.0, color="#92d050", alpha=0.8)
    ax.axhline(200, ls="--", lw=1.0, color="#ff6600", alpha=0.8)
    ax.text(hourly.index[0], 102, " Satisfactory (100)", fontsize=8, color="#5a8a2a")
    ax.text(hourly.index[0], 202, " Poor (200)", fontsize=8, color="#cc5500")

    ax.set_title("National AQI Trend (Hourly Mean vs. 24h Rolling)", pad=14)
    ax.set_ylabel("AQI (0–500)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", frameon=True)
    fig.autofmt_xdate(rotation=30)
    sns.despine()
    return _save(fig, out_dir / "05_aqi_time_series.png")


# --------------------------------------------------------------------------- #
# Orchestrator — the contract run_live.py depends on
# --------------------------------------------------------------------------- #
def generate_charts(master_df: pd.DataFrame,
                    out_dir: Path = ASSETS_DIR) -> List[Path]:
    """
    Generate all 5 dashboards. Individual failures are logged but do not
    abort the pipeline (so a missing optional column never blocks a run).
    Returns list of successfully saved paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if master_df is None or master_df.empty:
        logger.warning("generate_charts received an empty dataframe.")
        return []

    charts: List[Path] = []
    plan = [
        ("correlation_heatmap", plot_correlation_heatmap),
        ("top_cities_dot", plot_top_cities_dot),
        ("aqi_category_distribution", plot_aqi_category_distribution),
        ("pollutant_dominance_stacked", plot_pollutant_dominance_stacked),
        ("aqi_time_series", plot_aqi_time_series),
    ]
    for name, fn in plan:
        try:
            path = fn(master_df, out_dir=out_dir)
            if path is not None:
                charts.append(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Chart '%s' failed: %s", name, exc, exc_info=True)

    logger.info("generate_charts produced %d/%d charts.", len(charts), len(plan))
    return charts