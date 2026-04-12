"""
Master orchestrator — full pipeline execution order:

  EDA → Baseline TF-IDF → Advanced ML LDA → Deep Learning SBERT K-Means
        → GDELT Processor → GDELT Analysis
              → Event Impact Scoring (SBERT, uses DL rupture weeks)
                    → Visualise Results

Note: GDELT runs after Deep Learning so that event_impact_scoring.py can
      reference the semantic velocity / rupture output from the DL model.
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
    print("  Phase-1 pipeline complete. Outputs in reports/")
    print("=" * 60)
