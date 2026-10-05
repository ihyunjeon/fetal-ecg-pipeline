# Fetal ECG Extraction Pipeline

1D U-Net that extracts fetal ECG from abdominal electrode recordings.
Trained and evaluated on the PhysioNet Abdominal and Direct Fetal ECG Database.

## Setup

```bash
conda activate ml_dev
pip install torch mne wfdb scipy matplotlib numpy
```

## Data

Place `.edf` files from the database into `data/`:

```
data/r01.edf
data/r04.edf
data/r07.edf
data/r08.edf
data/r10.edf
```

## Usage

Run `pipeline.ipynb` end-to-end. It will:

1. Visualize raw abdominal + scalp signals with fetal QRS annotations
2. Load and preprocess data (bandpass filter, z-score, sliding windows)
3. Train a 1D U-Net with a hybrid MSE + correlation loss
4. Evaluate on held-out patient r10 (MSE, MAE, Pearson r)
5. Clinical validation: compare predicted vs. true fetal heart rate (BPM)
6. Frequency-domain comparison (power spectral density)

## Files

- `utils.py` — data loading, preprocessing, visualization
- `pipeline.ipynb` — full training and evaluation pipeline
