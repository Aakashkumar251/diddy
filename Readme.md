# IEEE SB GEHU ML Challenge – Fault Detection

This project builds a machine learning model to detect faulty device states from sensor measurements.

## Features

- Advanced feature engineering
- Spike detection
- Robust scaling
- LightGBM classifier
- Threshold optimization for best F1 score

## Dataset

47 sensor features (F01–F47)

Target:
0 → Normal  
1 → Faulty

## How to Run

Install libraries:

pip install pandas numpy scikit-learn lightgbm

Run:

python generate_predictions.py

## Output

The script generates:

FINAL.csv

Format:

ID,CLASS
1,0
2,1
3,0
