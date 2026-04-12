"""
Master orchestrator — full pipeline execution order:

  EDA → Baseline TF-IDF → Advanced ML LDA → Deep Learning SBERT K-Means
        → GDELT Processor (live fetch → latest 15-min snapshot)
              → GDELT Analysis
                    → Event Impact Scoring (SBERT, uses DL rupture weeks)
                          → Visualise Results

GDELT live feed: gdelt_processor.py calls gdelt_fetcher.py which downloads the
most recent GKG snapshot from http://data.gdeltproject.org/gdeltv2/lastupdate.txt
(updated every 15 minutes). Pass --local to skip the network and use local files.
"""
import os
import subprocess
import sys


def run_script(script_path: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  Running: {script_path}")
    print('─' * 60)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[STDERR]\n{result.stderr}")


SCRIPTS = [
    'src/eda.py',
    'src/baseline.py',
    'src/advanced_ml.py',
    'src/deep_learning.py',       # SBERT K-Means → rupture weeks
    'src/gdelt_processor.py',     # parse raw GDELT
    'src/gdelt_analysis.py',      # theme/tone summary
    'src/event_impact_scoring.py',# SBERT S_I using DL rupture output
    'src/visualize_results.py',
]

if __name__ == "__main__":
    os.makedirs('reports', exist_ok=True)
    for script in SCRIPTS:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"[SKIP] {script} not found")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Outputs in reports/")
    print("  GDELT data fetched live from the GDELT 2.0 API.")
    print("=" * 60)
