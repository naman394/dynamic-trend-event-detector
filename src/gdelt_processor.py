"""
GDELT Processor
===============
Parses GDELT 2.0 GKG data into a clean analysis-ready CSV.

Data source strategy (in priority order):
  1. Live fetch from GDELT API  (always freshest — updates every 15 min)
  2. Local .gkg.csv file in data/  (fallback if offline)
  3. Existing gdelt_processed.csv  (last-resort cache)

Run with --live to force a fresh download (default behaviour).
Run with --local to skip the network and use whatever is on disk.
"""

import os
import sys
import pandas as pd

DATA_DIR   = 'data'
OUTPUT_CSV = 'data/gdelt_processed.csv'

GKG_COLUMNS = [
    'GKGRECORDID', 'DATE', 'SOURCECOLLECTIONID', 'SOURCECOMMONNAME',
    'DOCUMENTIDENTIFIER', 'COUNTS', 'V2COUNTS', 'THEMES', 'V2THEMES',
    'LOCATIONS', 'V2LOCATIONS', 'PERSONS', 'V2PERSONS', 'ORGANIZATIONS',
    'V2ORGANIZATIONS', 'TONE', 'V2TONE', 'ENHANCEDDATES', 'GCAM',
    'SHARINGIMAGE', 'RELATEDIMAGES', 'SOCIALIMAGEEMBEDS',
    'SOCIALVIDEOEMBEDS', 'QUOTATIONS', 'ALLNAMES', 'AMOUNTS',
    'TRANSLATIONINFO', 'EXTRASXML'
]


def _extract_themes(v2themes: str) -> list:
    if pd.isna(v2themes) or not v2themes:
        return []
    return [item.split(',')[0] for item in v2themes.split(';') if item]


def _extract_tone(tone_str: str) -> float:
    if pd.isna(tone_str) or not tone_str:
        return 0.0
    try:
        return float(str(tone_str).split(',')[0])
    except ValueError:
        return 0.0


def process_gdelt(file_path: str) -> pd.DataFrame | None:
    """Parse a raw .gkg.csv (tab-separated) file into a clean DataFrame."""
    print(f"Processing GDELT file: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t', names=GKG_COLUMNS,
                         encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Error reading TSV: {e}")
        return None

    df['theme_list'] = df['V2THEMES'].apply(_extract_themes)
    df['tone_value'] = df['TONE'].apply(_extract_tone)
    print(f"Extracted {len(df):,} GDELT records.")
    return df[['GKGRECORDID', 'DATE', 'SOURCECOMMONNAME',
               'DOCUMENTIDENTIFIER', 'theme_list', 'tone_value']]


if __name__ == "__main__":
    live = '--local' not in sys.argv   # default: always try live

    processed_df = None

    # ── 1. Live fetch (default) ───────────────────────────────────────────────
    if live:
        try:
            from src.gdelt_fetcher import fetch_latest_gkg
            processed_df = fetch_latest_gkg(save_to=OUTPUT_CSV)
        except ImportError:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from src.gdelt_fetcher import fetch_latest_gkg
                processed_df = fetch_latest_gkg(save_to=OUTPUT_CSV)
            except Exception as e:
                print(f"Live fetch failed: {e}. Falling back to local file.")

    # ── 2. Local .gkg.csv fallback ────────────────────────────────────────────
    if processed_df is None:
        gkg_files = sorted(
            [f for f in os.listdir(DATA_DIR) if f.endswith('.gkg.csv')],
            reverse=True   # most recent first (filenames are timestamps)
        )
        if gkg_files:
            processed_df = process_gdelt(os.path.join(DATA_DIR, gkg_files[0]))
            if processed_df is not None:
                processed_df.to_csv(OUTPUT_CSV, index=False)
                print(f"Saved processed GDELT data to {OUTPUT_CSV}")
        else:
            print("No GDELT GKG files found in data/")

    # ── 3. Last-resort: existing processed CSV ─────────────────────────────────
    if processed_df is None and os.path.exists(OUTPUT_CSV):
        print(f"Using existing {OUTPUT_CSV} (may be stale).")
        processed_df = pd.read_csv(OUTPUT_CSV)

    if processed_df is not None:
        print(f"\nGDELT processing complete: {len(processed_df):,} records → {OUTPUT_CSV}")
    else:
        print("ERROR: No GDELT data available.")
