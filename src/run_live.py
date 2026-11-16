"""
src/run_live.py
---------------
Live pipeline orchestrator. Preserves the validated old `src/run.py` flow:
  1. fetch_aqi_snapshot()             -> dict contract
  2. If data_source == "API_FAILED"   -> sys.exit(0)  (keeps CI green)
  3. clean_and_update_master(dict)    -> master DataFrame
  4. generate_charts(master_df)       -> Seaborn dashboards
  5. generate_summary_excel(master_df)-> executive_summary.xlsx
  6. Print one-line summary

Works under BOTH:
    python src/run_live.py
    python -m src.run_live
"""
import sys
from pathlib import Path

# sys.path[0] is <root>/src when invoked as `python src/run_live.py`.
# Prepend <root> so `from src.* import ...` resolves under both invocation modes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Eager imports — all modules exist in the platform; no graceful degradation needed.
from src.ingestion.fetch_live import fetch_aqi_snapshot
from src.processing.cleaner_live import clean_and_update_master
from src.analytics.visualization import generate_charts
from src.analytics.reporting import generate_summary_excel


def main() -> None:
    # 1. Fetch
    raw_result = fetch_aqi_snapshot()

    # 2. CI-safe exit: API failed in cloud -> skip this hour, keep 100% live data
    if raw_result.get("data_source") == "API_FAILED":
        print("[WARN] Data source failed (API timeout). Skipping this hour "
              "to maintain 100% pure live data.")
        sys.exit(0)

    # 3. Clean + persist
    master_df = clean_and_update_master(raw_result)

    # 4. Visualize
    generate_charts(master_df)

    # 5. Excel report
    generate_summary_excel(master_df)

    print(
        f"[OK] Pipeline complete | Master rows: {len(master_df)} | "
        f"Source: {raw_result['data_source']} | API calls: {raw_result['api_calls_made']}"
    )


if __name__ == "__main__":
    main()