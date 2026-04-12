"""
GDELT Live Fetcher
==================
GDELT 2.0 publishes a new Global Knowledge Graph (GKG) file every 15 minutes.
This module always fetches the freshest available data instead of using a
static snapshot.

API endpoints used
------------------
  Latest 3 files (GKG + Events + Mentions):
      http://data.gdeltproject.org/gdeltv2/lastupdate.txt

  Last N 15-min windows (full archive index):
      http://data.gdeltproject.org/gdeltv2/masterfilelist.txt   (entire history)

  Individual file:
      http://data.gdeltproject.org/gdeltv2/<TIMESTAMP>.gkg.csv.zip

Usage
-----
    from src.gdelt_fetcher import fetch_latest_gkg, fetch_last_n_gkg

    # Single most-recent file (15-min window)
    df = fetch_latest_gkg()

    # Last 4 hours of coverage (16 × 15-min windows)
    df = fetch_last_n_gkg(n=16)
"""

import io
import os
import zipfile
import requests
import pandas as pd

LAST_UPDATE_URL  = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
MASTER_LIST_URL  = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
CACHE_DIR        = "data/gdelt_cache"
TIMEOUT          = 30   # seconds per HTTP request

GKG_COLUMNS = [
    'GKGRECORDID', 'DATE', 'SOURCECOLLECTIONID', 'SOURCECOMMONNAME',
    'DOCUMENTIDENTIFIER', 'COUNTS', 'V2COUNTS', 'THEMES', 'V2THEMES',
    'LOCATIONS', 'V2LOCATIONS', 'PERSONS', 'V2PERSONS', 'ORGANIZATIONS',
    'V2ORGANIZATIONS', 'TONE', 'V2TONE', 'ENHANCEDDATES', 'GCAM',
    'SHARINGIMAGE', 'RELATEDIMAGES', 'SOCIALIMAGEEMBEDS',
    'SOCIALVIDEOEMBEDS', 'QUOTATIONS', 'ALLNAMES', 'AMOUNTS',
    'TRANSLATIONINFO', 'EXTRASXML'
]


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _parse_gkg_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Unzip in-memory and parse the GKG TSV into a clean DataFrame."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith('.gkg.csv'))
        with zf.open(csv_name) as f:
            raw = pd.read_csv(f, sep='\t', names=GKG_COLUMNS,
                              encoding='utf-8', on_bad_lines='skip',
                              low_memory=False)
    raw['theme_list']  = raw['V2THEMES'].apply(_extract_themes)
    raw['tone_value']  = raw['TONE'].apply(_extract_tone)
    return raw[['GKGRECORDID', 'DATE', 'SOURCECOMMONNAME',
                'DOCUMENTIDENTIFIER', 'theme_list', 'tone_value']]


def _download_url(url: str, use_cache: bool = True) -> bytes:
    """Download a ZIP URL, optionally caching to disk."""
    fname = url.split('/')[-1]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, fname)

    if use_cache and os.path.exists(cache_path):
        print(f"  [cache] {fname}")
        with open(cache_path, 'rb') as f:
            return f.read()

    print(f"  [download] {fname} …", end=' ', flush=True)
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.content
    print(f"{len(data)//1024} KB")

    if use_cache:
        with open(cache_path, 'wb') as f:
            f.write(data)
    return data


# ── public API ───────────────────────────────────────────────────────────────

def get_latest_gkg_url() -> str:
    """
    Query GDELT's lastupdate.txt and return the URL of the newest GKG file.
    The file has 3 lines (GKG, Events, Mentions); the GKG line ends with .gkg.csv.zip
    """
    resp = requests.get(LAST_UPDATE_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    for line in resp.text.strip().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].endswith('.gkg.csv.zip'):
            return parts[2]
    raise RuntimeError("Could not find GKG URL in lastupdate.txt")


def fetch_latest_gkg(save_to: str = "data/gdelt_processed.csv") -> pd.DataFrame:
    """
    Download the most recent 15-min GKG snapshot, parse it, and optionally
    save to disk.

    Returns a DataFrame ready for gdelt_analysis.py / event_impact_scoring.py.
    """
    print("Fetching latest GDELT GKG snapshot …")
    url  = get_latest_gkg_url()
    timestamp = url.split('/')[-1].replace('.gkg.csv.zip', '')
    print(f"  Latest snapshot: {timestamp}")

    data = _download_url(url, use_cache=True)
    df   = _parse_gkg_zip(data)
    print(f"  Parsed {len(df):,} records from snapshot {timestamp}")

    if save_to:
        df.to_csv(save_to, index=False)
        print(f"  Saved → {save_to}")
    return df


def fetch_last_n_gkg(n: int = 16,
                     save_to: str = "data/gdelt_processed.csv") -> pd.DataFrame:
    """
    Fetch the last N × 15-min GKG windows and concatenate them.

    n=1   → latest 15 min
    n=4   → last 1 hour
    n=16  → last 4 hours   (default — good balance of recency vs volume)
    n=96  → last 24 hours

    Returns combined DataFrame saved to save_to.
    """
    print(f"Fetching last {n} GDELT GKG snapshots ({n*15} minutes of coverage) …")

    # Pull the master file list (tab-separated: size | hash | url)
    resp = requests.get(MASTER_LIST_URL, timeout=60)
    resp.raise_for_status()

    gkg_urls = [
        line.split()[2]
        for line in resp.text.strip().splitlines()
        if len(line.split()) == 3 and line.split()[2].endswith('.gkg.csv.zip')
    ]

    if not gkg_urls:
        raise RuntimeError("No GKG URLs found in masterfilelist.txt")

    # Most-recent N files (list is chronological, take last N)
    selected = gkg_urls[-n:]
    print(f"  Time range: {selected[0].split('/')[-1][:14]}  →  "
          f"{selected[-1].split('/')[-1][:14]}")

    frames = []
    for url in selected:
        try:
            data = _download_url(url, use_cache=True)
            frames.append(_parse_gkg_zip(data))
        except Exception as e:
            print(f"  [SKIP] {url.split('/')[-1]} — {e}")

    if not frames:
        raise RuntimeError("No GKG files could be downloaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} records from {len(frames)} snapshots")

    if save_to:
        combined.to_csv(save_to, index=False)
        print(f"  Saved → {save_to}")

    return combined


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch live GDELT GKG data")
    parser.add_argument('--windows', type=int, default=1,
                        help="Number of 15-min windows to fetch (default=1 = latest only)")
    parser.add_argument('--out', default="data/gdelt_processed.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if args.windows == 1:
        fetch_latest_gkg(save_to=args.out)
    else:
        fetch_last_n_gkg(n=args.windows, save_to=args.out)
