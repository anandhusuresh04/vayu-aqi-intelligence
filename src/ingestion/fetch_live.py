"""
src/ingestion/fetch_live.py
----------------------------
Live CPCB fetcher for data.gov.in resource 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.

Preserved contracts (from validated working implementation):
  * Returns dict: {records, data_source, api_calls_made, records_retrieved}
  * data_source ∈ {"LIVE_API", "SYNTHETIC_SEED", "API_FAILED"}
  * Local + missing key  -> SYNTHETIC_SEED fallback
  * CI  (GITHUB_ACTIONS) -> return API_FAILED with records=[] (orchestrator exits 0)
  * Local + live failure -> re-raise (never contaminate with synthetic data)
  * Pagination advances by len(all_records), NOT offset += page_limit
  * Explicit `"error" in payload` check (data.gov.in returns 200 + error body)
  * Only `User-Agent` header is sent (validated as sufficient)
  * Snapshot filename: data/raw/live/aqi_live_YYYYMMDD_HHMMSS.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Path bootstrap: correct whether invoked via `python src/run_live.py`
# (sys.path[0] = <root>/src) or `python -m src.run_live` (sys.path[0] = <root>).
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# .env lives at project root; load explicitly so it works from any CWD
load_dotenv(PROJECT_ROOT / ".env")

RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "live"
FALLBACK_SEED_PATH = RAW_DIR / "fallback_historical_seed.json"

SYNTHETIC_PROVENANCE = "SYNTHETIC_SEED"
SYNTHETIC_DESC = "SYNTHETIC_SEED (rng=42, modeled on typical Indian urban profiles)"


# --------------------------------------------------------------------------- #
# Synthetic seed (local-dev only; generated on first run if missing)
# --------------------------------------------------------------------------- #
def generate_seed_if_missing(seed_path: Path = FALLBACK_SEED_PATH) -> None:
    """Create a deterministic synthetic seed if none exists. Local-dev convenience."""
    if seed_path.exists():
        return
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    stations = [
        ("Delhi", "Delhi", "Anand Vihar, Delhi - DPCC", "28.6476", "77.3158", 165.0),
        ("Maharashtra", "Mumbai", "Bandra, Mumbai - MPCB", "19.0627", "72.8461", 68.0),
        ("Kerala", "Kochi", "Vyttila, Kochi - Kerala PCB", "9.9674", "76.3182", 34.0),
        ("Karnataka", "Bengaluru", "BTM Layout, Bengaluru - KSPCB", "12.9135", "77.5988", 42.0),
        ("Tamil Nadu", "Chennai", "Alandur Bus Depot, Chennai - CPCB", "12.9971", "80.1915", 51.0),
        ("West Bengal", "Kolkata", "Victoria, Kolkata - WBPCB", "22.5448", "88.3426", 95.0),
        ("Uttar Pradesh", "Lucknow", "Lalbagh, Lucknow - UPPCB", "26.8467", "80.9462", 130.0),
        ("", "Byrnihat", "Central Academy, Byrnihat - PCBA", "26.0420", "91.8720", 145.0),
    ]
    start_dt = datetime(2026, 11, 8, 0, 0, 0)
    records: List[Dict[str, Any]] = []
    for hour_offset in range(7 * 24):
        current_dt = start_dt + timedelta(hours=hour_offset)
        ts_str = current_dt.strftime("%d-%m-%Y %H:%M:%S")
        diurnal_factor = 1.25 if current_dt.hour in (7, 8, 9, 19, 20, 21) else 0.85
        for state, city, station, lat, lon, base_pm25 in stations:
            pm25 = max(8.0, rng.normal(base_pm25 * diurnal_factor, base_pm25 * 0.12))
            pm10 = pm25 * rng.uniform(1.65, 2.15) + rng.normal(15.0, 5.0)
            no2 = pm25 * 0.42 + rng.normal(18.0, 4.0)
            so2 = pm25 * 0.11 + rng.normal(9.0, 2.0)
            co_mg = round(max(0.2, pm25 * 0.011 + rng.normal(0.45, 0.08)), 2)
            ozone = max(10.0, 55.0 * (1.4 if 11 <= current_dt.hour <= 16 else 0.7) + rng.normal(0, 5.0))
            nh3 = max(5.0, pm25 * 0.18 + rng.normal(12.0, 3.0))
            pollutant_map = {
                "PM2.5": round(pm25, 1), "PM10": round(pm10, 1), "NO2": round(no2, 1),
                "SO2": round(so2, 1), "CO": co_mg, "OZONE": round(ozone, 1), "NH3": round(nh3, 1),
            }
            for pol_id, val in pollutant_map.items():
                is_na = rng.random() < 0.06
                avg_str = "NA" if is_na else str(val)
                min_str = "NA" if is_na else str(round(val * 0.75, 2))
                max_str = "NA" if is_na else str(round(val * 1.25, 2))
                records.append({
                    "country": "India", "state": state, "city": city, "station": station,
                    "last_update": ts_str, "latitude": lat, "longitude": lon,
                    "pollutant_id": pol_id,
                    "pollutant_min": min_str, "pollutant_max": max_str, "pollutant_avg": avg_str,
                })
    payload = {
        "index_name": RESOURCE_ID, "title": SYNTHETIC_DESC, "provenance": SYNTHETIC_PROVENANCE,
        "total": len(records), "count": len(records), "records": records,
    }
    with open(seed_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------- #
# Public API — preserves the dict contract the cleaner + orchestrator expect
# --------------------------------------------------------------------------- #
def fetch_aqi_snapshot(page_limit: int = 500) -> Dict[str, Any]:
    """
    Fetch the current CPCB snapshot.

    Returns dict with keys:
      records           : List[Dict]  (raw API records OR synthetic seed records)
      data_source       : "LIVE_API" | "SYNTHETIC_SEED" | "API_FAILED"
      api_calls_made    : int
      records_retrieved : int

    Behavior matrix:
      key missing/wrong + local   -> SYNTHETIC_SEED
      key missing/wrong + CI      -> SYNTHETIC_SEED  (matches old behavior)
      key OK + live OK            -> LIVE_API
      key OK + live fails + local -> raise (no contamination)
      key OK + live fails + CI    -> API_FAILED with empty records (run exits 0)
      live returns 0 records      -> SYNTHETIC_SEED (matches old behavior)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    generate_seed_if_missing(FALLBACK_SEED_PATH)

    api_key = os.getenv("DATA_GOV_API_KEY", "").strip()
    all_records: List[Dict[str, Any]] = []
    api_calls_made = 0

    # --- Local dev path: no key -> synthetic seed -----------------------------
    if not api_key or api_key == "your_registered_data_gov_in_api_key_here":
        print("[WARN] No API key provided. Using synthetic seed.")
        with open(FALLBACK_SEED_PATH, "r", encoding="utf-8") as f:
            seed_payload = json.load(f)
        return {
            "records": seed_payload.get("records", []),
            "data_source": SYNTHETIC_PROVENANCE,
            "api_calls_made": 0,
            "records_retrieved": len(seed_payload.get("records", [])),
        }

    # --- Live API path -------------------------------------------------------
    offset = 0
    total_expected: int | None = None
    MAX_PAGES = 50
    # Only UA is sent — validated as sufficient; do NOT add extra headers.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        for _ in range(MAX_PAGES):
            params = {"api-key": api_key, "format": "json",
                      "limit": page_limit, "offset": offset}
            api_calls_made += 1
            print(f"[INFO] Fetching live CPCB page (offset={offset}, limit={page_limit})...")

            resp = requests.get(BASE_URL, params=params, headers=headers, timeout=45)

            # data.gov.in returns HTML on WAF blocks or JSON errors on 200 OK
            try:
                payload = resp.json()
            except json.JSONDecodeError:
                print(f"[ERROR] Non-JSON response. First 200 chars: {resp.text[:200]}")
                resp.raise_for_status()
                raise

            # CRITICAL: expired-key responses look like {"error": "..."} with HTTP 200
            if "error" in payload:
                raise ValueError(f"API Error: {payload['error']}")

            batch = payload.get("records", [])
            all_records.extend(batch)

            if total_expected is None:
                total_expected = int(payload.get("total") or payload.get("count") or len(batch))

            if len(all_records) >= total_expected or len(batch) == 0:
                break

            # CRITICAL: advance by what we actually received, not by page_limit
            offset = len(all_records)
            time.sleep(0.3)
        else:
            print(f"[WARN] Hit MAX_PAGES; got {len(all_records)}/{total_expected} records.")

        if len(all_records) > 0:
            ts_label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_file = RAW_DIR / f"aqi_live_{ts_label}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"total": total_expected, "records": all_records}, f, indent=2)

            return {
                "records": all_records,
                "data_source": "LIVE_API",
                "api_calls_made": api_calls_made,
                "records_retrieved": len(all_records),
            }

    except Exception as exc:
        print(f"[WARN] Live API fetch failed ({exc}).")

        # CI: exit cleanly (orchestrator sees API_FAILED and does sys.exit(0))
        if os.getenv("GITHUB_ACTIONS") == "true":
            print("[INFO] GitHub Actions detected. Skipping synthetic fallback to keep data 100% live.")
            return {
                "records": [],
                "data_source": "API_FAILED",
                "api_calls_made": api_calls_made,
                "records_retrieved": 0,
            }

        # Local: hard-fail. Never contaminate with synthetic data silently.
        print("[ERROR] Local API fetch failed. Check API key / connectivity.")
        print("[ERROR] Halting pipeline to prevent synthetic data contamination.")
        raise

    # Fallback: API returned 200 with zero records (rare, but possible)
    print("[WARN] API returned 0 records. Falling back to synthetic seed.")
    with open(FALLBACK_SEED_PATH, "r", encoding="utf-8") as f:
        seed_payload = json.load(f)
    return {
        "records": seed_payload.get("records", []),
        "data_source": SYNTHETIC_PROVENANCE,
        "api_calls_made": api_calls_made,
        "records_retrieved": len(seed_payload.get("records", [])),
    }


if __name__ == "__main__":
    result = fetch_aqi_snapshot()
    print(f"\n[OK] data_source={result['data_source']} "
          f"records={result['records_retrieved']} api_calls={result['api_calls_made']}")