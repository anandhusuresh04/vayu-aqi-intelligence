"""
src/analytics/reporting.py
--------------------------
Builds assets/executive_summary.xlsx — a 5-sheet snapshot.

Public API (contract with src/run_live.py):
    generate_summary_excel(master_df) -> Path

Sheets:
  Top_Cities_AQI        : City, State, Mean AQI, Max AQI
  State_Summary         : State, Stations Reporting, Total Observations,
                          Mean AQI, PM2.5 Missing Count
  AQI_Distribution      : Category, Count
  Pollutant_Correlations: correlation matrix (pollutants)
  Audit_Log             : data/processed/audit_log.csv (empty-safe)
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

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "audit_log.csv"
EXCEL_NAME = "executive_summary.xlsx"

logger = logging.getLogger(__name__)

POLLUTANT_NUMERIC_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]
AQI_CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very_Poor", "Severe"]


# --------------------------------------------------------------------------- #
# Sheet builders
# --------------------------------------------------------------------------- #
def _sheet_top_cities(df: pd.DataFrame) -> pd.DataFrame:
    required = ["city", "state", "Overall_AQI"]
    if not all(c in df.columns for c in required):
        logger.warning("Top_Cities_AQI: missing required columns; writing empty.")
        return pd.DataFrame(columns=["City", "State", "Mean AQI", "Max AQI"])

    g = (
        df.dropna(subset=["Overall_AQI"])
          .groupby(["city", "state"], as_index=False)
          .agg(**{"Mean AQI": ("Overall_AQI", "mean"),
                  "Max AQI": ("Overall_AQI", "max")})
          .sort_values("Mean AQI", ascending=False)
    )
    g = g.rename(columns={"city": "City", "state": "State"})
    g["Mean AQI"] = g["Mean AQI"].round(1)
    g["Max AQI"] = g["Max AQI"].round(1)
    return g[["City", "State", "Mean AQI", "Max AQI"]].reset_index(drop=True)


def _sheet_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ["state", "station"]
    if not all(c in df.columns for c in required):
        logger.warning("State_Summary: missing required columns; writing empty.")
        return pd.DataFrame(columns=[
            "State", "Stations Reporting", "Total Observations",
            "Mean AQI", "PM2.5 Missing Count",
        ])

    # stations reporting = unique station names per state
    agg = df.groupby("state").agg(
        **{
            "Stations Reporting": ("station", "nunique"),
            "Total Observations": ("station", "size"),
        }
    )
    # Mean AQI (guard missing Overall_AQI column)
    if "Overall_AQI" in df.columns:
        agg["Mean AQI"] = df.groupby("state")["Overall_AQI"].mean().round(1)
    else:
        agg["Mean AQI"] = np.nan
    # PM2.5 missing count
    if "PM2.5" in df.columns:
        agg["PM2.5 Missing Count"] = df.groupby("state")["PM2.5"].apply(
            lambda s: int(s.isna().sum())
        )
    else:
        agg["PM2.5 Missing Count"] = 0

    out = agg.reset_index().rename(columns={"state": "State"})
    out = out.sort_values("Mean AQI", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def _sheet_aqi_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "AQI_Bucket" not in df.columns:
        logger.warning("AQI_Distribution: AQI_Bucket missing; writing empty.")
        return pd.DataFrame({"Category": AQI_CATEGORY_ORDER, "Count": [0] * len(AQI_CATEGORY_ORDER)})
    counts = (
        df["AQI_Bucket"].value_counts()
          .reindex(AQI_CATEGORY_ORDER, fill_value=0)
    )
    return pd.DataFrame({"Category": counts.index, "Count": counts.values.astype(int)})


def _sheet_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in POLLUTANT_NUMERIC_COLS if c in df.columns]
    if len(cols) < 2:
        logger.warning("Pollutant_Correlations: <2 pollutant columns; writing empty.")
        return pd.DataFrame()
    corr = df[cols].corr(numeric_only=True).round(3)
    corr.index.name = "Pollutant"
    return corr.reset_index()


def _sheet_audit() -> pd.DataFrame:
    if not AUDIT_CSV_PATH.exists():
        logger.warning("Audit_Log: %s not found; writing empty sheet.", AUDIT_CSV_PATH)
        return pd.DataFrame(columns=[
            "run_timestamp_utc", "data_source", "api_calls_made",
            "records_retrieved", "raw_records_count", "pivoted_stations_count",
            "valid_aqi_stations", "pm25_missing_pct_this_run",
            "unmapped_state_rows", "new_rows_appended",
        ])
    try:
        return pd.read_csv(AUDIT_CSV_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Audit_Log read failed: %s", exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_summary_excel(master_df: pd.DataFrame,
                           out_dir: Path = ASSETS_DIR) -> Path:
    """
    Build assets/executive_summary.xlsx with the 5 required sheets.
    Returns the saved path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / EXCEL_NAME

    if master_df is None or master_df.empty:
        logger.warning("generate_summary_excel received empty df; writing empty workbook.")
        master_df = pd.DataFrame()

    sheets = {
        "Top_Cities_AQI": _sheet_top_cities(master_df),
        "State_Summary": _sheet_state_summary(master_df),
        "AQI_Distribution": _sheet_aqi_distribution(master_df),
        "Pollutant_Correlations": _sheet_correlations(master_df),
        "Audit_Log": _sheet_audit(),
    }

    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                # NaN-safe write so Excel shows blanks, not "nan"
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                _autosize_columns(writer.sheets[sheet_name], frame)
        logger.info("Saved executive summary -> %s", out_path)
    except Exception as exc:
        logger.exception("Failed to write Excel report: %s", exc)
        raise

    return out_path


# --------------------------------------------------------------------------- #
# Excel polish (column widths)
# --------------------------------------------------------------------------- #
def _autosize_columns(ws, frame: pd.DataFrame, min_width: int = 10, max_width: int = 42) -> None:
    from openpyxl.utils import get_column_letter
    if frame is None or frame.empty:
        return
    for i, col in enumerate(frame.columns, start=1):
        header_len = len(str(col))
        sample_len = int(frame[col].astype(str).str.len().head(50).max()) if col in frame else 0
        width = max(min_width, min(max_width, max(header_len, sample_len) + 3))
        ws.column_dimensions[get_column_letter(i)].width = width


if __name__ == "__main__":
    # Smoke test on the live master if it exists
    master_path = PROJECT_ROOT / "data" / "processed" / "live_master.csv"
    if master_path.exists():
        df = pd.read_csv(master_path)
        p = generate_summary_excel(df)
        print(f"[OK] wrote {p}")
    else:
        print(f"[SKIP] {master_path} not found — run the pipeline first.")