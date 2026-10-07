"""Rebuild every artifact from scratch, in the correct order.

    python -m src.pipeline              # full rebuild (~5-10 min)
    python -m src.pipeline --from train # reuse the existing dataset

The research experiments (src/experiments/run_experiments.py) are NOT part of this:
they justify the settings in src/config.py and take much longer.
"""

import subprocess
import sys

from src.config import BASE_DIR

STEPS = [
    ("generate", "src.data_generation.motor_digital_twin_v5_10"),
    ("preprocess", "src.preprocessing.preprocess_dataset"),
    ("eda", "src.eda.eda_analysis"),
    ("train", "src.classification.fault_classifier"),
    ("anomaly", "src.anomaly_detection.isolation_forest"),
]

if __name__ == "__main__":
    names = [n for n, _ in STEPS]
    start = names.index(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
    for name, module in STEPS[start:]:
        print(f"\n=== {name}: python -m {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], cwd=BASE_DIR, check=True)
    print("\nPipeline complete. Launch the dashboard with: streamlit run app.py")
