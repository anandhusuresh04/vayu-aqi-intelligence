"""
src/processing/feature_eng.py
-----------------------------
Feature engineering for both live and historical ML pipelines.

Public API:
    create_features(df) -> pd.DataFrame

Adds (idempotently, on a copy — the caller's frame is never mutated):
  * PM25_PM10_Ratio : PM2.5 / PM10  (NaN when PM10 is 0 or missing)
  * Is_Weekend      : bool, True for Sat/Sun
  * Season          : Winter | Summer | Monsoon | Post-Monsoon (Indian seasons)

Robust to:
  * timestamp_utc as str or datetime, tz-aware or naive
  * missing PM2.5 / PM10 columns
  * re-running (existing columns are overwritten, not duplicated)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# Indian meteorological seasons by month
#   Winter       : Dec, Jan, Feb
#   Summer       : Mar, Apr, May  (pre-monsoon)
#   Monsoon      : Jun, Jul, Aug, Sep
#   Post-Monsoon : Oct, Nov
SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    """Coerce to tz-aware UTC datetime; return NaT on failures."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt


def _resolve_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Find the first available timestamp column."""
    for col in ("timestamp_utc", "timestamp_ist", "last_update", "timestamp"):
        if col in df.columns:
            return col
    return None


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with PM25_PM10_Ratio, Is_Weekend, Season added.
    Never mutates the input frame.
    """
    if df is None:
        raise ValueError("create_features received None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"create_features expected DataFrame, got {type(df)}")

    out = df.copy()

    # ---------- PM25_PM10_Ratio ----------
    if "PM2.5" in out.columns and "PM10" in out.columns:
        pm25 = pd.to_numeric(out["PM2.5"], errors="coerce")
        pm10 = pd.to_numeric(out["PM10"], errors="coerce")
        # Avoid divide-by-zero producing inf; leave NaN instead
        ratio = pm25 / pm10.replace({0: np.nan})
        out["PM25_PM10_Ratio"] = ratio.round(4)
    else:
        logger.warning("PM2.5 or PM10 column missing — PM25_PM10_Ratio will be NaN.")
        out["PM25_PM10_Ratio"] = np.nan

    # ---------- Timestamp-dependent features ----------
    ts_col = _resolve_timestamp_column(out)
    if ts_col is None:
        logger.warning("No timestamp column found — Is_Weekend/Season will be NaN.")
        out["Is_Weekend"] = pd.Series([pd.NA] * len(out), dtype="boolean")
        out["Season"] = pd.Series([pd.NA] * len(out), dtype="object")
        return out

    ts = _to_datetime_utc(out[ts_col])

    # Is_Weekend — weekday(): Monday=0 ... Sunday=6
    is_weekend = ts.dt.weekday.isin([5, 6])
    # Preserve missingness as pd.NA in the nullable Boolean dtype
    out["Is_Weekend"] = is_weekend.astype("boolean")
    out.loc[ts.isna(), "Is_Weekend"] = pd.NA

    # Season — based on IST month for correct Indian-season alignment
    try:
        month_ist = ts.dt.tz_convert("Asia/Kolkata").dt.month
    except (TypeError, AttributeError):
        month_ist = ts.dt.month
    out["Season"] = month_ist.map(SEASON_BY_MONTH).astype("object")
    out.loc[ts.isna(), "Season"] = pd.NA

    logger.info(
        "create_features: added PM25_PM10_Ratio, Is_Weekend, Season (ts col=%s).",
        ts_col,
    )
    return out


if __name__ == "__main__":
    # Smoke test
    sample = pd.DataFrame({
        "PM2.5": [60.0, 120.0, np.nan],
        "PM10":  [120.0, 0.0, 200.0],
        "timestamp_utc": ["2025-01-15T10:00:00Z", "2025-07-20T15:00:00Z", "2025-11-02T08:00:00Z"],
    })
    print(create_features(sample))