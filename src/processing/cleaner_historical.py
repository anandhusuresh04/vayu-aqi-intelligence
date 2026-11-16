"""
src/processing/cleaner_historical.py
------------------------------------
Normalize the raw combined historical CSV into the ML-ready
data/processed/historical_ml.csv.

Public API:
    clean_historical(raw_source=None, output_path=None, co_mode="mgm3")
        -> pd.DataFrame
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Path bootstrap
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the live cleaner's CPCB logic instead of duplicating it
from src.processing.cleaner_live import (  # noqa: E402
    CITY_TO_STATE,
    classify_aqi_bucket,
    compute_station_overall_aqi,
)

HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
DEFAULT_RAW = HISTORICAL_DIR / "historical_raw_combined.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROCESSED_DIR / "historical_ml.csv"

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Column aliasing
# --------------------------------------------------------------------------- #
def _norm_key(s: str) -> str:
    """
    Alphanumeric-only, lowercased.
    CRITICAL FIX: Strips units in parentheses first.
    'PM2.5 (µg/m³)' -> 'pm25'
    'TOT-RF (mm)'   -> 'totrf'
    """
    s_clean = re.sub(r"\(.*?\)", "", str(s))
    return re.sub(r"[^a-z0-9]", "", s_clean.lower())


# Every key here MUST be the output of _norm_key().
COLUMN_ALIASES: Dict[str, str] = {
    # Primary pollutants
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "no": "NOx",          # Added to catch 'NO (µg/m³)'
    "nox": "NOx",
    "so2": "SO2",
    "co": "CO",
    "carbonmonoxide": "CO",
    "o3": "OZONE",
    "ozone": "OZONE",
    "nh3": "NH3",
    "ammonia": "NH3",
    "nitrogendioxide": "NO2",
    "sulfurdioxide": "SO2",
    "sulphurdioxide": "SO2",
    # VOCs
    "benzene": "Benzene",
    "toluene": "Toluene",
    "xylene": "Xylene",
    "oxylene": "Xylene",      # Added
    "mpxylene": "Xylene",     # Added
    "ethbenzene": "Benzene",  # Added
    # AQI targets
    "aqi": "AQI",
    "aqibucket": "AQI_Bucket",
    "aqicategory": "AQI_Bucket",
    "bucket": "AQI_Bucket",
    # Location
    "state": "state",
    "city": "city",
    "station": "station",
    "sitename": "station",
    "site": "station",
    "latitude": "latitude",
    "longitude": "longitude",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
    # Timestamp
    "date": "timestamp",
    "datetime": "timestamp",
    "timestamp": "timestamp",
    "lastupdate": "timestamp",
    # Meteorology
    "at": "AT",
    "ambienttemperature": "AT",
    "temperature": "AT",
    "temp": "AT",
    "bp": "BP",
    "barometricpressure": "BP",
    "pressure": "BP",
    "rh": "RH",
    "relativehumidity": "RH",
    "humidity": "RH",
    "ws": "WS",
    "windspeed": "WS",
    "wd": "WD",
    "winddirection": "WD",
    "sr": "SR",
    "solarradiation": "SR",
    "rf": "Rain",
    "totrf": "Rain",          # Added to catch 'TOT-RF (mm)'
    "rain": "Rain",
    "rainfall": "Rain",
    "precipitation": "Rain",
    # Provenance
    "sourcefile": "source_file",
    "sourceyear": "source_year",
    "year": "source_year",
}

# Canonical numeric columns to coerce
NUMERIC_CANONICAL: List[str] = [
    "PM2.5", "PM10", "NO2", "NOx", "SO2", "CO", "OZONE", "NH3",
    "Benzene", "Toluene", "Xylene", "AQI",
    "AT", "BP", "RH", "WS", "WD", "SR", "Rain",
    "latitude", "longitude",
]

# Pollutant columns used for the "drop rows with zero measurements" step
POLLUTANT_COLUMNS: List[str] = [
    "PM2.5", "PM10", "NO2", "NOx", "SO2", "CO", "OZONE", "NH3",
    "Benzene", "Toluene", "Xylene",
]

# Strings that should be treated as missing
NA_STRINGS = {"", "NA", "N/A", "na", "n/a", "nan", "NaN", "None", "null", "-", "--"}

# Final column order in the output CSV
FINAL_COLUMN_ORDER: List[str] = [
    "state", "city", "station", "latitude", "longitude",
    "timestamp_utc", "timestamp_ist", "source_year", "source_file",
    "PM2.5", "PM10", "NO2", "NOx", "SO2", "CO", "OZONE", "NH3",
    "Benzene", "Toluene", "Xylene",
    "AT", "BP", "RH", "WS", "WD", "SR", "Rain",
    "AQI", "AQI_Bucket",
]


# --------------------------------------------------------------------------- #
# Stage 1: load
# --------------------------------------------------------------------------- #
def _load_raw(raw_source: Union[None, str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(raw_source, pd.DataFrame):
        return raw_source.copy()
    if raw_source is None:
        raw_source = DEFAULT_RAW
    p = Path(raw_source)
    if not p.exists():
        raise FileNotFoundError(
            f"Raw historical CSV not found: {p}. "
            "Run `python -m src.ingestion.fetch_historical` first."
        )
    logger.info("Loading raw historical: %s", p)
    return pd.read_csv(p, dtype=str, low_memory=False)


# --------------------------------------------------------------------------- #
# Stage 2: normalize columns
# --------------------------------------------------------------------------- #
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    unmatched: List[str] = []
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(_norm_key(col))
        if canonical:
            rename_map[col] = canonical
        else:
            unmatched.append(col)

    if unmatched:
        logger.warning(
            "Dropping %s unrecognized column(s): %s",
            len(unmatched), unmatched[:20] + (["..."] if len(unmatched) > 20 else []),
        )

    df = df.rename(columns=rename_map)
    # Drop unrecognized columns
    df = df[[c for c in df.columns if c in COLUMN_ALIASES.values()]]

    # Coalesce any duplicate canonical columns (first non-null wins)
    if df.columns.duplicated().any():
        result: Dict[str, pd.Series] = {}
        for col in df.columns.unique():
            subset = df.loc[:, df.columns == col].copy()
            subset = subset.replace(list(NA_STRINGS), np.nan)
            result[col] = subset.bfill(axis=1).iloc[:, 0]
        df = pd.DataFrame(result, index=df.index)
        logger.info("Coalesced duplicate source columns into canonical names.")

    logger.info("Normalized columns -> %s", list(df.columns))
    return df


# --------------------------------------------------------------------------- #
# Stage 3: timestamps
# --------------------------------------------------------------------------- #
def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        logger.warning("No timestamp column found — writing NaT for all rows.")
        df["timestamp_utc"] = pd.NA
        df["timestamp_ist"] = pd.NA
        return df

    raw = df["timestamp"].astype(str).str.strip()

    ts = pd.to_datetime(raw, errors="coerce")
    n_na_first = ts.isna().sum()

    if n_na_first > 0.5 * len(ts):
        ts2 = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        if ts2.notna().sum() > ts.notna().sum():
            logger.info(
                "Switched to dayfirst=True parsing (%s -> %s successfully parsed).",
                ts.notna().sum(), ts2.notna().sum(),
            )
            ts = ts2

    # Normalize to Asia/Kolkata then UTC
    try:
        if ts.notna().any() and getattr(ts.dt, "tz", None) is None:
            ts_ist = ts.dt.tz_localize(
                "Asia/Kolkata", ambiguous="NaT", nonexistent="NaT"
            )
        elif ts.notna().any():
            ts_ist = ts.dt.tz_convert("Asia/Kolkata")
        else:
            ts_ist = ts
    except (TypeError, AttributeError):
        ts_ist = ts

    df["timestamp_ist"] = ts_ist.dt.strftime("%Y-%m-%d %H:%M:%S%z") if hasattr(ts_ist, "dt") else pd.NA

    try:
        ts_utc = ts_ist.dt.tz_convert("UTC") if getattr(ts_ist.dt, "tz", None) else ts_ist
    except (TypeError, AttributeError):
        ts_utc = ts_ist
    df["timestamp_utc"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(ts_utc, "dt") else pd.NA

    df = df.drop(columns=["timestamp"])
    n_bad = int((df["timestamp_utc"] == "").sum() + df["timestamp_utc"].isna().sum())
    logger.info(
        "Parsed timestamps: %s rows OK, %s rows NaT.",
        len(df) - n_bad, n_bad,
    )
    return df


# --------------------------------------------------------------------------- #
# Stage 4: location standardization
# --------------------------------------------------------------------------- #
def _clean_str_series(s: pd.Series, titlecase: bool = False) -> pd.Series:
    out = s.astype(str).str.strip()
    if titlecase:
        out = out.str.title()
    out = out.mask(out.isin(["nan", "Nan", "NaN", "none", "None", ""]), np.nan)
    return out


def _standardize_location(df: pd.DataFrame) -> pd.DataFrame:
    if "city" in df.columns:
        df["city"] = _clean_str_series(df["city"], titlecase=True)
    if "state" in df.columns:
        df["state"] = _clean_str_series(df["state"], titlecase=True)
        df["state"] = df["state"].str.replace("_", " ", regex=False)
    if "station" in df.columns:
        df["station"] = _clean_str_series(df["station"], titlecase=False)

    # Fill missing state via CITY_TO_STATE, else UNKNOWN
    if "state" in df.columns and "city" in df.columns:
        missing_mask = df["state"].isna()
        df.loc[missing_mask, "state"] = (
            df.loc[missing_mask, "city"].map(CITY_TO_STATE).fillna("UNKNOWN")
        )
    elif "state" not in df.columns:
        df["state"] = "UNKNOWN"

    return df


# --------------------------------------------------------------------------- #
# Stage 5: numeric coercion
# --------------------------------------------------------------------------- #
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_CANONICAL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Stage 6: ensure AQI
# --------------------------------------------------------------------------- #
def _ensure_aqi(df: pd.DataFrame, co_mode: str = "mgm3") -> pd.DataFrame:
    data_source = "SYNTHETIC_SEED" if co_mode == "mgm3" else "LIVE_API"

    needs_aqi = "AQI" not in df.columns or df["AQI"].isna().all()
    if needs_aqi:
        logger.info(
            "AQI column missing — computing from sub-indices for %s rows "
            "(this may take several minutes on large data)...",
            len(df),
        )
        # Apply is slow, but necessary for the complex breakpoint logic
        df["AQI"] = df.apply(
            lambda r: compute_station_overall_aqi(r, data_source=data_source),
            axis=1,
        )
    else:
        logger.info("AQI already present — trusting source values.")

    if "AQI_Bucket" not in df.columns or df["AQI_Bucket"].isna().all():
        df["AQI_Bucket"] = df["AQI"].apply(classify_aqi_bucket)

    return df


# --------------------------------------------------------------------------- #
# Stage 7: drop empty rows
# --------------------------------------------------------------------------- #
def _drop_empty_pollutant_rows(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in POLLUTANT_COLUMNS if c in df.columns]
    if not present:
        logger.warning("No pollutant columns present — cannot drop empty rows.")
        return df
    mask = df[present].isna().all(axis=1)
    n_drop = int(mask.sum())
    if n_drop:
        logger.info("Dropping %s rows with zero usable pollutants.", n_drop)
    return df.loc[~mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 8: dedupe
# --------------------------------------------------------------------------- #
def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["state", "city", "station", "timestamp_utc"] if c in df.columns]
    if not keys:
        logger.warning("Dedupe skipped — no usable key columns.")
        return df
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    logger.info("Deduped %s -> %s (dropped %s)", before, len(df), before - len(df))
    return df


# --------------------------------------------------------------------------- #
# Stage 9: reorder
# --------------------------------------------------------------------------- #
def _reorder(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in FINAL_COLUMN_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    return df[present + extra]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def clean_historical(
    raw_source: Union[None, str, Path, pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
    co_mode: str = "mgm3",
) -> pd.DataFrame:
    """Full historical cleaning pipeline. Writes historical_ml.csv; returns df."""
    if co_mode not in {"mgm3", "subindex"}:
        raise ValueError(f"co_mode must be 'mgm3' or 'subindex', got {co_mode!r}")

    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_raw(raw_source)
    logger.info("Loaded raw: %s rows x %s cols.", len(df), df.shape[1])

    df = _normalize_columns(df)
    df = _parse_timestamps(df)
    df = _standardize_location(df)
    df = _coerce_numeric(df)
    df = _ensure_aqi(df, co_mode=co_mode)
    df = _drop_empty_pollutant_rows(df)
    df = _dedupe(df)
    df = _reorder(df)

    if df.empty:
        raise ValueError("Cleaning produced an empty dataframe — refusing to write.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved historical_ml -> %s (%s rows x %s cols)",
        output_path, len(df), df.shape[1],
    )

    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Clean historical AQI -> historical_ml.csv")
    parser.add_argument("--raw", type=Path, default=None,
                        help="Path to raw combined CSV (default: data/raw/historical/historical_raw_combined.csv)")
    parser.add_argument("--co-mode", choices=["mgm3", "subindex"], default="mgm3",
                        help="CO encoding in source: mg/m3 (default) or 0-500 sub-index")
    args = parser.parse_args()

    df = clean_historical(raw_source=args.raw, co_mode=args.co_mode)
    print(f"\n[OK] historical_ml.csv: {len(df)} rows x {df.shape[1]} cols")
    print(f"     columns: {list(df.columns)}")
    if "AQI" in df.columns:
        print(f"     AQI range: [{df['AQI'].min():.1f}, {df['AQI'].max():.1f}] "
              f"(NaN={df['AQI'].isna().sum()})")
    if "AQI_Bucket" in df.columns:
        print(f"     bucket counts: {df['AQI_Bucket'].value_counts().to_dict()}")