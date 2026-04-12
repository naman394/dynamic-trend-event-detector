#!/bin/bash
#
# refresh_gdelt.sh — Fetch fresh GDELT data and re-run analysis
#
# Usage:
#   ./refresh_gdelt.sh           # fetch latest 15-min snapshot
#   ./refresh_gdelt.sh 4         # fetch last 4 snapshots (1 hour)
#   ./refresh_gdelt.sh 16        # fetch last 16 snapshots (4 hours)
#   ./refresh_gdelt.sh 96        # fetch last 96 snapshots (24 hours)
#
# What it does:
#   1. Clears GDELT cache (forces fresh download)
#   2. Fetches live data from GDELT API
#   3. Re-runs GDELT analysis (top themes, tone)
#   4. Re-runs GDELT verification + SBERT impact scoring
#   5. Shows summary of results
#

set -e
cd "$(dirname "$0")"

WINDOWS=${1:-1}
VENV="venv/bin/activate"

echo "═══════════════════════════════════════════════════════════"
echo "  GDELT Live Refresh — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Fetching ${WINDOWS} snapshot(s) ($(( WINDOWS * 15 )) minutes of coverage)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Activate virtual environment
source "$VENV"

# Step 1: Clear cache for fresh data
if [ -d "data/gdelt_cache" ]; then
    rm -rf data/gdelt_cache
    echo "✓ Cache cleared"
fi

# Step 2: Fetch live GDELT data
echo ""
if [ "$WINDOWS" -eq 1 ]; then
    python3 -c "
import sys; sys.path.insert(0,'.')
from src.gdelt_fetcher import fetch_latest_gkg
df = fetch_latest_gkg(save_to='data/gdelt_processed.csv')
print(f'✓ Fetched {len(df):,} records — snapshot {df[\"DATE\"].iloc[0]}')
"
else
    python3 -c "
import sys; sys.path.insert(0,'.')
from src.gdelt_fetcher import fetch_last_n_gkg
df = fetch_last_n_gkg(n=${WINDOWS}, save_to='data/gdelt_processed.csv')
print(f'✓ Fetched {len(df):,} records from ${WINDOWS} snapshots')
"
fi

# Step 3: Re-run analysis notebooks
echo ""
echo "Re-running GDELT analysis …"
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=dtdetector_venv \
  --ExecutePreprocessor.timeout=300 \
  notebooks/05_gdelt_analysis.ipynb 2>&1 | tail -1

echo "Re-running GDELT verification + impact scoring …"
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=dtdetector_venv \
  --ExecutePreprocessor.timeout=300 \
  notebooks/09_gdelt_verification_impact.ipynb 2>&1 | tail -1

# Step 4: Quick summary
echo ""
echo "═══════════════════════════════════════════════════════════"
python3 -c "
import pandas as pd
from ast import literal_eval
from collections import Counter

df = pd.read_csv('data/gdelt_processed.csv')
print(f'  Records     : {len(df):,}')
print(f'  Timestamp   : {df[\"DATE\"].iloc[0]}')
print(f'  Sources     : {df[\"SOURCECOMMONNAME\"].nunique()} unique')
print(f'  Mean tone   : {df[\"tone_value\"].mean():.2f}')
print()

all_t = []
for t in df['theme_list']:
    try: all_t.extend(literal_eval(t) if isinstance(t,str) else t)
    except: pass
print('  Top 10 themes:')
for theme, cnt in Counter(all_t).most_common(10):
    print(f'    {theme:<45} {cnt}')
print()
print(f'  Impact scores → reports/event_impact_scores.csv')
print(f'  Theme chart   → reports/gdelt_theme_tone.png')
"
echo "═══════════════════════════════════════════════════════════"
echo "  Done! Open notebooks/05 or 09 in Jupyter to see charts."
echo "═══════════════════════════════════════════════════════════"
