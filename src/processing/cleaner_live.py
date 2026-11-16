"""
src/processing/cleaner_live.py
------------------------------
Transforms a fetch_aqi_snapshot() dict into the canonical master CSV.

Preserved contracts (from validated working implementation):
  * Entrypoint: clean_and_update_master(fetch_result: dict) -> pd.DataFrame
  * Writes to data/processed/live_master.csv (rename from old aqi_master.csv
    is a blueprint change; update MASTER_CSV_PATH constant if your notebooks
    still reference aqi_master.csv).
  * CO logic threads `data_source`:
      - LIVE_API       -> CO arrives as 0-500 sub-index; pass through capped at 500.
      - SYNTHETIC_SEED -> CO arrives in mg/m³; apply CPCB mg/m³ breakpoints.
    `CO_mg_m3` is produced in BOTH cases (via invert_co_subindex_to_mgm3).
  * Validity gate: ≥3 valid sub-indices AND (PM2.5 or PM10) present.
    Failing stations are KEPT with Overall_AQI=NaN, AQI_Bucket="Insufficient_Data".
  * CITY_TO_STATE fallback with "UNKNOWN" retention (Byrnihat intentionally
    omitted from the map).
  * Timestamp parsing: explicit "%d-%m-%Y %H:%M:%S" -> Asia/Kolkata -> UTC.
  * Dedupe on [state, city, station, timestamp_utc].
  * Audit log: exact 10-column schema.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MASTER_CSV_PATH = PROCESSED_DIR / "live_master.csv"   # blueprint name
AUDIT_CSV_PATH = PROCESSED_DIR / "audit_log.csv"

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]

CITY_TO_STATE = {
    "Delhi": "Delhi", "New Delhi": "Delhi",
    "Mumbai": "Maharashtra", "Navi Mumbai": "Maharashtra",
    "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Kochi": "Kerala", "Ernakulam": "Kerala", "Thiruvananthapuram": "Kerala",
    "Kozhikode": "Kerala", "Thrissur": "Kerala", "Kollam": "Kerala",
    "Eloor": "Kerala", "Kannur": "Kerala",
    "Bengaluru": "Karnataka", "Mysuru": "Karnataka", "Mangaluru": "Karnataka",
    "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu", "Madurai": "Tamil Nadu",
    "Kolkata": "West Bengal", "Howrah": "West Bengal", "Asansol": "West Bengal",
    "Hyderabad": "Telangana",
    "Visakhapatnam": "Andhra Pradesh", "Vijayawada": "Andhra Pradesh",
    "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh", "Noida": "Uttar Pradesh",
    "Greater Noida": "Uttar Pradesh", "Ghaziabad": "Uttar Pradesh",
    "Agra": "Uttar Pradesh", "Varanasi": "Uttar Pradesh",
    "Gurugram": "Haryana", "Faridabad": "Haryana",
    "Patna": "Bihar", "Muzaffarpur": "Bihar",
    "Ahmedabad": "Gujarat", "Gandhinagar": "Gujarat",
    "Surat": "Gujarat", "Vadodara": "Gujarat",
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan",
    "Kota": "Rajasthan", "Udaipur": "Rajasthan",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh",
    "Gwalior": "Madhya Pradesh", "Jabalpur": "Madhya Pradesh",
    "Chandigarh": "Chandigarh",
    "Bhubaneswar": "Odisha",
    "Guwahati": "Assam",
    # "Byrnihat" intentionally omitted: must surface as state="UNKNOWN"
    # to exercise the audit path.
}

# Official CPCB NAQI breakpoints. Tuples: (B_LO, B_HI, I_LO, I_HI)
# Units: µg/m³ for PM2.5, PM10, NO2, SO2, OZONE, NH3 | mg/m³ for CO
CPCB_BREAKPOINTS = {
    "PM2.5": [(0.0, 30.0, 0, 50), (30.0, 60.0, 51, 100), (60.0, 90.0, 101, 200),
              (90.0, 120.0, 201, 300), (120.0, 250.0, 301, 400), (250.0, 500.0, 401, 500)],
    "PM10":  [(0.0, 50.0, 0, 50), (50.0, 100.0, 51, 100), (100.0, 250.0, 101, 200),
              (250.0, 350.0, 201, 300), (350.0, 430.0, 301, 400), (430.0, 600.0, 401, 500)],
    "NO2":   [(0.0, 40.0, 0, 50), (40.0, 80.0, 51, 100), (80.0, 180.0, 101, 200),
              (180.0, 280.0, 201, 300), (280.0, 400.0, 301, 400), (400.0, 600.0, 401, 500)],
    "SO2":   [(0.0, 40.0, 0, 50), (40.0, 80.0, 51, 100), (80.0, 380.0, 101, 200),
              (380.0, 800.0, 201, 300), (800.0, 1600.0, 301, 400), (1600.0, 2400.0, 401, 500)],
    "CO":    [(0.0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10.0, 101, 200),
              (10.0, 17.0, 201, 300), (17.0, 34.0, 301, 400), (34.0, 50.0, 401, 500)],
    "OZONE": [(0.0, 50.0, 0, 50), (50.0, 100.0, 51, 100), (100.0, 168.0, 101, 200),
              (168.0, 208.0, 201, 300), (208.0, 748.0, 301, 400), (748.0, 1000.0, 401, 500)],
    "NH3":   [(0.0, 200.0, 0, 50), (200.0, 400.0, 51, 100), (400.0, 800.0, 101, 200),
              (800.0, 1200.0, 201, 300), (1200.0, 1800.0, 301, 400), (1800.0, 2400.0, 401, 500)],
}


def compute_sub_index(pollutant: str, conc: float, data_source: str = "LIVE_API") -> float:
    """
    CPCB sub-index I_p ∈ [0, 500] for one pollutant reading.

    CO SPECIAL CASE:
      LIVE_API publishes CO already as a 0-500 sub-index -> pass through capped at 500.
      SYNTHETIC_SEED publishes CO in mg/m³ -> apply the mg/m³ breakpoint table.
    """
    if pd.isna(conc) or conc < 0 or pollutant not in CPCB_BREAKPOINTS:
        return np.nan

    conc_val = float(conc)

    if data_source == "LIVE_API" and pollutant == "CO":
        return round(min(max(conc_val, 0.0), 500.0), 1)

    table = CPCB_BREAKPOINTS[pollutant]
    for b_lo, b_hi, i_lo, i_hi in table:
        if b_lo <= conc_val <= b_hi:
            sub_idx = ((i_hi - i_lo) / (b_hi - b_lo)) * (conc_val - b_lo) + i_lo
            return round(float(sub_idx), 1)

    if conc_val > table[-1][1]:
        return 500.0
    return np.nan


def invert_co_subindex_to_mgm3(co_val: float, data_source: str = "LIVE_API") -> float:
    """
    Produce CO_mg_m3 (physical units) regardless of source encoding.
      LIVE_API       -> invert the 0-500 sub-index back to mg/m³.
      SYNTHETIC_SEED -> already mg/m³; return as-is.
    """
    if pd.isna(co_val) or co_val < 0:
        return np.nan
    val = float(co_val)
    if data_source != "LIVE_API":
        return round(val, 3)

    for b_lo, b_hi, i_lo, i_hi in CPCB_BREAKPOINTS["CO"]:
        if i_lo <= val <= i_hi:
            conc = ((val - i_lo) / (i_hi - i_lo)) * (b_hi - b_lo) + b_lo
            return round(float(conc), 3)
    if val > 500:
        return 50.0
    return np.nan


def classify_aqi_bucket(aqi_val: float) -> str:
    if pd.isna(aqi_val):
        return "Insufficient_Data"
    if aqi_val <= 50:
        return "Good"
    if aqi_val <= 100:
        return "Satisfactory"
    if aqi_val <= 200:
        return "Moderate"
    if aqi_val <= 300:
        return "Poor"
    if aqi_val <= 400:
        return "Very_Poor"
    return "Severe"


def compute_station_overall_aqi(row: pd.Series, data_source: str = "LIVE_API") -> float:
    """
    Station AQI = max of valid sub-indices, subject to the CPCB validity gate:
      - at least 3 valid sub-indices
      - at least one of PM2.5 or PM10 present
    Returns NaN when the gate fails (row is kept, not dropped).
    """
    sub_indices = {
        pol: compute_sub_index(pol, row.get(pol, np.nan), data_source=data_source)
        for pol in POLLUTANTS
    }
    valid = {k: v for k, v in sub_indices.items() if pd.notna(v)}

    has_particulate = ("PM2.5" in valid) or ("PM10" in valid)
    if len(valid) >= 3 and has_particulate:
        return round(float(max(valid.values())), 1)
    return np.nan


def clean_and_update_master(fetch_result: Dict) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = fetch_result.get("records", [])
    if not records:
        raise ValueError("clean_and_update_master received empty records; aborting.")

    data_source = fetch_result["data_source"]
    api_calls_made = fetch_result["api_calls_made"]
    records_retrieved = fetch_result["records_retrieved"]

    raw_df = pd.DataFrame(records)
    raw_count = len(raw_df)

    # Column rename from API schema -> canonical (per architecture spec)
    raw_df = raw_df.rename(columns={
        "min_value": "pollutant_min",
        "max_value": "pollutant_max",
        "avg_value": "pollutant_avg",
    })

    # 1. Timestamp: "%d-%m-%Y %H:%M:%S" is IST -> localize Asia/Kolkata -> UTC
    parsed_ist = pd.to_datetime(
        raw_df["last_update"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    parsed_ist = parsed_ist.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
    raw_df["timestamp_ist"] = parsed_ist.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    raw_df["timestamp_utc"] = parsed_ist.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 2. State: CITY_TO_STATE fallback; unmapped -> "UNKNOWN" (rows retained)
    raw_df["city"] = raw_df["city"].astype(str).str.strip()
    if "state" not in raw_df.columns:
        raw_df["state"] = ""
    raw_df["state"] = (
        raw_df["state"].fillna("").astype(str).str.strip().str.replace("_", " ")
    )
    missing_state_mask = raw_df["state"] == ""
    raw_df.loc[missing_state_mask, "state"] = (
        raw_df.loc[missing_state_mask, "city"].map(CITY_TO_STATE).fillna("UNKNOWN")
    )

    # 3. "NA"/"null"/etc. -> NaN, then to numeric
    raw_df["pollutant_avg_num"] = pd.to_numeric(
        raw_df["pollutant_avg"].replace(["NA", "na", "null", "None", ""], np.nan),
        errors="coerce",
    )

    # 4. Pivot long -> wide (one row per station-timestamp)
    index_cols = [
        "country", "state", "city", "station", "latitude", "longitude",
        "timestamp_utc", "timestamp_ist",
    ]
    pivoted = (
        raw_df.pivot_table(
            index=index_cols, columns="pollutant_id",
            values="pollutant_avg_num", aggfunc="mean",
        ).reset_index()
    )
    pivoted.columns.name = None

    for pol in POLLUTANTS:
        if pol not in pivoted.columns:
            pivoted[pol] = np.nan

    # 5. Sub-indices + CO_mg_m3 + Overall_AQI
    for pol in POLLUTANTS:
        pivoted[f"{pol}_subindex"] = pivoted[pol].apply(
            lambda x, p=pol: compute_sub_index(p, x, data_source=data_source)
        )

    pivoted["CO_mg_m3"] = pivoted["CO"].apply(
        lambda x: invert_co_subindex_to_mgm3(x, data_source=data_source)
    )

    pivoted["Overall_AQI"] = pivoted.apply(
        compute_station_overall_aqi, axis=1, data_source=data_source
    )
    pivoted["AQI_Bucket"] = pivoted["Overall_AQI"].apply(classify_aqi_bucket)
    pivoted["data_source"] = data_source

    # 6. Dedupe + append
    dedupe_cols = ["state", "city", "station", "timestamp_utc"]
    if MASTER_CSV_PATH.exists():
        existing_df = pd.read_csv(MASTER_CSV_PATH)
        before_dedup = len(existing_df)
        combined_df = pd.concat([existing_df, pivoted], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=dedupe_cols, keep="last")
        new_rows_appended = len(combined_df) - before_dedup
    else:
        combined_df = pivoted.drop_duplicates(subset=dedupe_cols, keep="last")
        new_rows_appended = len(combined_df)

    combined_df = combined_df.sort_values(
        ["timestamp_utc", "state", "city", "station"]
    ).reset_index(drop=True)
    combined_df.to_csv(MASTER_CSV_PATH, index=False)

    # 7. Audit log (exact 10-column schema)
    unmapped_state_rows = int((pivoted["state"] == "UNKNOWN").sum())
    pm25_missing_pct_this_run = round(float(pivoted["PM2.5"].isna().mean() * 100), 2)
    valid_aqi_stations = int(pivoted["Overall_AQI"].notna().sum())

    audit_row = pd.DataFrame([{
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": data_source,
        "api_calls_made": api_calls_made,
        "records_retrieved": records_retrieved,
        "raw_records_count": raw_count,
        "pivoted_stations_count": len(pivoted),
        "valid_aqi_stations": valid_aqi_stations,
        "pm25_missing_pct_this_run": pm25_missing_pct_this_run,
        "unmapped_state_rows": unmapped_state_rows,
        "new_rows_appended": max(0, new_rows_appended),
    }])

    if AUDIT_CSV_PATH.exists():
        audit_row.to_csv(AUDIT_CSV_PATH, mode="a", header=False, index=False)
    else:
        audit_row.to_csv(AUDIT_CSV_PATH, mode="w", header=True, index=False)

    return combined_df