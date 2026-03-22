import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n--- Running {script_path} ---")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors in {script_path}:\n{result.stderr}")

scripts = [
    'src/eda.py',
    'src/baseline.py',
    'src/advanced_ml.py',
    'src/deep_learning.py',
    'src/hybrid_temporal.py',
    'src/gdelt_processor.py',
    'src/gdelt_analysis.py',
    'src/event_impact_scoring.py'
]

if __name__ == "__main__":
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Skipping {script}, file not found.")

    print("\n--- Pipeline execution complete. Check 'reports/' for outputs. ---")
