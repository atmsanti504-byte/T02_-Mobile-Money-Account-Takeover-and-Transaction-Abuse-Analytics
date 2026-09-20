#!/usr/bin/env bash
# Runs the full Milestone 2 prototype pipeline end to end.
set -e
cd "$(dirname "$0")"

echo "== 1/5 Generating synthetic data =="
python3 src/generate_data.py

echo "== 2/5 EDA + feature engineering =="
python3 src/eda_features.py

echo "== 3/5 Supervised + anomaly models =="
python3 src/models.py

echo "== 4/5 Text mining =="
python3 src/text_mining.py

echo "== 5/5 Simulation + predictive early-warning =="
python3 src/simulation.py
python3 src/predictive_risk.py

echo "== Running tests =="
python3 -m pytest tests/ -v

echo ""
echo "Pipeline complete. Open prototype/dashboard.html in a browser to view the analyst prototype."
