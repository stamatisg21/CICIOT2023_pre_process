# CICIOT2023_pre_process

A preprocessing pipeline for the CICIOT2023 cybersecurity dataset.

This repository loads raw train/test/validation CSV files, maps detailed attack labels into broader categories, balances the training set, scales numeric features, and saves cleaned datasets along with preprocessing artifacts.

You can find the dataset here:

```bash
https://www.kaggle.com/datasets/himadri07/ciciot2023/data
```

## Project structure

- `main.py` - data preparation and preprocessing pipeline
- `train/` - raw training data (`train.csv`)
- `test/` - raw test data (`test.csv`)
- `validation/` - raw validation data (`validation.csv`)
- `train_ready.csv` - cleaned and scaled training data
- `test_ready.csv` - cleaned and scaled test data
- `val_ready.csv` - cleaned and scaled validation data
- `requirements.txt` - Python dependencies
- `best_cnn_ids.keras` - saved Keras model (existing artifact)

## What it does

- loads raw CSVs from `train/`, `test/`, and `validation/`
- maps verbose attack labels to broader categories like `DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, `BruteForce`, and `Benign`
- cleans numeric data by replacing infinite values and imputing missing values with medians
- balances the training set by resampling each class to a target size
- standardizes features using `StandardScaler`
- saves cleaned datasets and preprocessing artifacts (`scaler.pkl`, `label_encoder.pkl`, `feature_names.pkl`)
- generates a class distribution plot at `class_distribution.png`

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `requirements.txt` includes packages required by `main.py`.

## Usage

Run the preprocessing pipeline:

```bash
python main.py
```

After running, the repository will contain:

- `train_ready.csv`
- `test_ready.csv`
- `val_ready.csv`
- `scaler.pkl`
- `label_encoder.pkl`
- `feature_names.pkl`
- `class_distribution.png`

## Notes

- `main.py` is designed to be run from the repository root.
- If the raw CSV files are missing, place them in the expected folders before running.
- The pipeline currently uses a fixed class balance target of 20,000 samples per class.
