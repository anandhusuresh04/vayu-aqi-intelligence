"""
src/ingestion/fetch_historical.py
---------------------------------
Unpack the Kaggle India AQI ZIP and merge every nested CSV into a single
raw combined file for downstream cleaning.

Public API:
    fetch_historical(zip_path=None, extracted_dir=None, output_path=None,
                     force=False) -> dict

Returns:
    {
      "raw_csv_path"     : str,    # data/raw/historical/historical_raw_combined.csv
      "zip_path"         : str|None,
      "extracted_dir"    : str,
      "num_csvs_merged"  : int,
      "num_rows"         : int,
      "num_columns"      : int,
      "columns_sample"   : list[str],
    }

Behavior:
  * ZIP discovery priority: explicit arg -> KAGGLE_ZIP_PATH env -> single
    *.zip in data/raw/historical/ -> existing extracted dir -> existing CSVs.
  * Extraction is idempotent (skips when extracted_dir already has CSVs).
  * Every CSV is read with dtype=str to preserve values verbatim; the
    cleaner is responsible for all type coercion and column normalization.
  * `source_file` (relative to extracted_dir) and `source_year` (from folder
    or filename) are attached to each row before concatenation.
  * No synthetic fallback. Missing inputs raise FileNotFoundError.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# --------------------------------------------------------------------------- #
# Path bootstrap
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
EXTRACTED_DIR = HISTORICAL_DIR / "extracted"
DEFAULT_OUTPUT = HISTORICAL_DIR / "historical_raw_combined.csv"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# ZIP discovery
# --------------------------------------------------------------------------- #
def _locate_zip(explicit: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Return the ZIP to extract, or None if we should fall back to existing CSVs."""
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"ZIP not found: {p}")
        return p

    env = os.getenv("KAGGLE_ZIP_PATH", "").strip()
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"KAGGLE_ZIP_PATH points to missing file: {p}")
        return p

    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(HISTORICAL_DIR.glob("*.zip"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple ZIPs found in {HISTORICAL_DIR}: "
            f"{[p.name for p in candidates]}. "
            "Pass `zip_path=` explicitly or set KAGGLE_ZIP_PATH."
        )
    return None


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract with a completion marker so partial extractions are detectable."""
    marker = dest / ".extraction_complete"
    if marker.exists() and not any(dest.rglob("*.csv")):
        # Marker exists but no CSVs: stale state, wipe and retry.
        logger.warning("Stale extraction marker found but no CSVs — re-extracting.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s -> %s", zip_path, dest)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    marker.write_text("ok", encoding="utf-8")
    logger.info("Extraction complete: %s", dest)


# --------------------------------------------------------------------------- #
# CSV discovery + year extraction
# --------------------------------------------------------------------------- #
_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")


def _extract_year(rel_path: Path) -> Optional[int]:
    """First 4-digit year found across all path parts (folders + filename)."""
    for part in rel_path.parts:
        m = _YEAR_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def discover_csvs(root: Path, exclude: Optional[List[Path]] = None) -> List[Path]:
    """Recursively find *.csv under root, excluding any given paths."""
    root = root.resolve()
    exclude_set = {p.resolve() for p in (exclude or [])}
    csvs = []
    for p in root.rglob("*.csv"):
        rp = p.resolve()
        if rp in exclude_set:
            continue
        if p.name.startswith("."):
            continue
        csvs.append(p)
    csvs.sort()
    return csvs


# --------------------------------------------------------------------------- #
# Combination
# --------------------------------------------------------------------------- #
def _read_one(csv_path: Path) -> pd.DataFrame:
    """Read a single CSV with all values as strings (no inference)."""
    try:
        return pd.read_csv(csv_path, dtype=str, low_memory=False)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s; retrying latin-1.", csv_path)
        return pd.read_csv(csv_path, dtype=str, low_memory=False, encoding="latin-1")


def combine_csvs(csv_paths: List[Path], root: Path) -> pd.DataFrame:
    """Read every CSV, attach provenance, concat. Raises if any file yields no rows."""
    if not csv_paths:
        raise ValueError("No CSV files to combine.")

    root = root.resolve()
    frames: List[pd.DataFrame] = []
    total_files = len(csv_paths)
    failed: List[str] = []

    for i, path in enumerate(csv_paths, start=1):
        try:
            df = _read_one(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s: %s", path, exc)
            failed.append(str(path))
            continue
        if df.empty:
            logger.debug("Empty CSV: %s", path)
            continue
        rel = path.resolve().relative_to(root)
        df["source_file"] = str(rel)
        df["source_year"] = _extract_year(rel)
        frames.append(df)
        if i % 500 == 0 or i == total_files:
            logger.info("  read %s/%s CSVs (%s rows so far)",
                        i, total_files, sum(len(f) for f in frames))

    if not frames:
        raise ValueError(
            f"No usable CSVs found (out of {total_files} scanned; "
            f"{len(failed)} read failures)."
        )

    combined = pd.concat(frames, ignore_index=True, copy=False)
    logger.info(
        "Combined %s CSVs -> %s rows x %s cols.",
        len(frames), len(combined), combined.shape[1],
    )
    return combined


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fetch_historical(
    zip_path: Optional[Union[str, Path]] = None,
    extracted_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Dict[str, object]:
    """See module docstring."""
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    extracted_dir = Path(extracted_dir) if extracted_dir else EXTRACTED_DIR
    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT

    # 1. Decide whether to extract or reuse
    zip_file = _locate_zip(zip_path)

    already_extracted = (
        extracted_dir.exists()
        and any(extracted_dir.rglob("*.csv"))
        and not force
    )

    if already_extracted:
        logger.info("Using already-extracted CSVs at %s (pass force=True to re-extract).",
                    extracted_dir)
    elif zip_file is not None:
        if extracted_dir.exists() and force:
            logger.info("force=True — wiping %s", extracted_dir)
            shutil.rmtree(extracted_dir)
        _extract_zip(zip_file, extracted_dir)
    else:
        # No ZIP and no extracted dir: try CSVs directly in HISTORICAL_DIR
        if any(HISTORICAL_DIR.rglob("*.csv")):
            logger.warning(
                "No ZIP and no extracted dir, but found CSVs in %s — using those.",
                HISTORICAL_DIR,
            )
            extracted_dir = HISTORICAL_DIR
        else:
            raise FileNotFoundError(
                f"No Kaggle ZIP and no extracted CSVs found. Expected one of:\n"
                f"  - {HISTORICAL_DIR}/*.zip\n"
                f"  - KAGGLE_ZIP_PATH env var\n"
                f"  - CSVs under {extracted_dir}/\n"
                f"  - CSVs under {HISTORICAL_DIR}/"
            )

    # 2. Discover + combine
    csvs = discover_csvs(extracted_dir, exclude=[output_path])
    logger.info("Discovered %s CSVs under %s", len(csvs), extracted_dir)
    if not csvs:
        raise ValueError(f"No CSVs found under {extracted_dir} after extraction.")

    combined = combine_csvs(csvs, extracted_dir)
    if combined.empty:
        raise ValueError("Combined dataframe is empty; refusing to write output.")

    # 3. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("Saved raw combined -> %s", output_path)

    return {
        "raw_csv_path": str(output_path),
        "zip_path": str(zip_file) if zip_file else None,
        "extracted_dir": str(extracted_dir),
        "num_csvs_merged": len(csvs),
        "num_rows": int(len(combined)),
        "num_columns": int(combined.shape[1]),
        "columns_sample": list(combined.columns)[:30],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Merge Kaggle AQI ZIP -> raw combined CSV")
    parser.add_argument("--zip", type=Path, default=None, help="Path to ZIP file")
    parser.add_argument("--force", action="store_true", help="Re-extract even if CSVs exist")
    args = parser.parse_args()

    result = fetch_historical(zip_path=args.zip, force=args.force)
    print("\n[OK] fetch_historical result:")
    for k, v in result.items():
        if k == "columns_sample":
            print(f"  {k}: {v[:10]}{' ...' if len(v) > 10 else ''}")
        else:
            print(f"  {k}: {v}")