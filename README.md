# PulmoGuard

**Selective-Prediction Chest X-Ray Triage System**

PulmoGuard is a pneumonia triage classifier that knows when *not* to answer.
Instead of reporting a single accuracy number on 100% of cases (the standard
but clinically misleading way most chest X-ray classifiers are evaluated),
PulmoGuard uses **Monte Carlo Dropout** to estimate its own predictive
uncertainty on every image and can **abstain** on cases it isn't confident
about, deferring them to a radiologist instead of guessing.

The headline result is a **risk-coverage curve**: model accuracy as a
function of how much of the test set it agrees to answer. A well-calibrated
triage tool should look something like:

| Coverage | Accuracy |
|----------|----------|
| 100% (answers everything) | ~92% |
| 80% (defers hardest 20%)  | ~97%+ |
| 50% (defers hardest 50%)  | ~99%+ |

This is the shape of a genuinely deployable clinical decision-support tool,
not a leaderboard number.

---

## Why this matters

A model that is confidently wrong is more dangerous in a clinical setting
than one that says "I'm not sure." Raw softmax scores from a standard
deterministic network are well known to be overconfident and poorly
calibrated. PulmoGuard addresses this directly:

1. **Monte Carlo Dropout** ([Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)) keeps dropout active at inference and runs multiple stochastic forward passes. Disagreement across passes approximates epistemic uncertainty.
2. **Predictive entropy** over the averaged prediction is used as the uncertainty score, normalized to `[0, 1]` for interpretability.
3. **Configurable abstention threshold** lets you trade off coverage vs. accuracy without retraining — tune it against the risk-coverage curve for your deployment's risk tolerance.
4. **Expected Calibration Error (ECE)** is reported as a secondary diagnostic.

---

## Repository structure

```
pulmoguard-cv/
├── configs/
│   └── config.yaml              # single source of truth for all hyperparameters
├── notebooks/
│   └── train_colab.ipynb        # run this in Colab (free T4) to train + evaluate
├── src/pulmoguard/
│   ├── data.py                  # dataset / dataloader construction
│   ├── model.py                 # EfficientNet-B0 with dropout-equipped head
│   ├── uncertainty.py           # MC-Dropout inference + entropy calculation
│   ├── train.py                 # training loop (AMP, checkpointing, early stopping)
│   ├── evaluate.py              # risk-coverage curve + calibration + plots
│   ├── infer.py                 # production inference class (used locally)
│   └── utils.py                 # seeding, device, config, logging
├── scripts/
│   ├── download_data.sh         # Kaggle dataset download helper
│   └── run_inference.py         # batch CLI inference over a folder of images
├── app/
│   └── serve.py                 # FastAPI server for local/production inference
├── tests/
│   └── test_model.py            # unit tests (no dataset required, fast)
├── outputs/                     # checkpoints, plots, metrics (gitignored)
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Workflow: train in Colab, run locally

This project is split by design: **train and evaluate on a free Colab T4
GPU**, then **download the trained checkpoint and run inference locally**
(CLI or API) — no GPU required for inference.

### Step 1 — Train in Colab

1. Push this repo to GitHub (or upload the zip and extract in Colab).
2. Open `notebooks/train_colab.ipynb` in Google Colab.
3. Set Runtime → Change runtime type → **T4 GPU**.
4. Run all cells. The notebook will:
   - Clone/mount the repo
   - Install dependencies
   - Download the dataset from Kaggle (you'll need a free Kaggle API token)
   - Train the model (`src/pulmoguard/train.py`)
   - Evaluate it and generate the risk-coverage curve (`src/pulmoguard/evaluate.py`)
   - Save the checkpoint and plots to Google Drive (survives Colab session disconnects)

Expected training time on a free-tier T4: **~30–50 minutes** for the default
8-epoch config with early stopping — comfortably within a single Colab
session.

### Step 2 — Download the trained checkpoint

From Google Drive (or Colab's file browser), download:
```
pulmoguard_best.pt
```
and place it locally at:
```
outputs/checkpoints/pulmoguard_best.pt
```

### Step 3 — Run inference locally

**Set up the environment:**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # installs the pulmoguard package in editable mode
```

**Option A — Batch CLI (folder of images → CSV):**
```bash
python scripts/run_inference.py \
    --checkpoint outputs/checkpoints/pulmoguard_best.pt \
    --image-dir /path/to/xray_images \
    --output predictions.csv
```

**Option B — REST API (FastAPI):**
```bash
uvicorn app.serve:app --host 0.0.0.0 --port 8000
```
Then, in another terminal:
```bash
curl -X POST -F "file=@sample_xray.jpeg" http://localhost:8000/predict
```

Example response:
```json
{
  "predicted_class": "PNEUMONIA",
  "confidence": 0.94,
  "normalized_entropy": 0.09,
  "abstain": false,
  "class_probabilities": {"NORMAL": 0.06, "PNEUMONIA": 0.94},
  "mc_dropout_passes": 20,
  "filename": "sample_xray.jpeg",
  "message": "Prediction confidence within accepted operating range."
}
```

**Option C — Python API directly:**
```python
from pulmoguard.infer import PulmoGuardPredictor

predictor = PulmoGuardPredictor(checkpoint_path="outputs/checkpoints/pulmoguard_best.pt")
result = predictor.predict("sample_xray.jpeg")
print(result)
```

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— 5,856 pediatric chest X-rays labeled NORMAL / PNEUMONIA, released under
CC BY 4.0. Download via:

```bash
pip install kaggle
# place your Kaggle API token at ~/.kaggle/kaggle.json first
bash scripts/download_data.sh data
```

**Note on the validation split:** the original Kaggle release ships only
16 images in `val/`, which is too small to reliably use for model selection
or threshold tuning. By default, `data.py` carves a stratified 15% validation
split out of `train/` instead (documented, not a silent hack) — see
`build_dataloaders(..., carve_val_from_train=True)`.

---

## Configuration

All hyperparameters live in `configs/config.yaml` — nothing is hardcoded
in the training/eval/inference code. Key knobs:

| Key | Purpose |
|---|---|
| `model.dropout_p` | Dropout rate, used both for regularization during training and as the MC-Dropout mechanism at inference |
| `uncertainty.mc_dropout_passes` | Number of stochastic forward passes at inference (accuracy/latency tradeoff) |
| `uncertainty.abstain_entropy_threshold` | Normalized entropy above which the system abstains — tune this against `outputs/plots/risk_coverage_curve.png` for your desired coverage/accuracy operating point |
| `train.epochs`, `train.lr`, `train.batch_size` | Standard training hyperparameters |

---

## Testing

```bash
pytest tests/ -v
```

Tests use synthetic tensors and require no dataset download, so they run in
seconds and are safe to run before spending Colab GPU time.

---

## Limitations & responsible use

- Trained on a single public pediatric dataset (Guangzhou Women and
  Children's Medical Center) — performance will likely degrade on adult
  populations, different scanner hardware, or different patient demographics
  without additional fine-tuning and validation.
- MC-Dropout approximates epistemic uncertainty but does not capture all
  failure modes (e.g. confidently-wrong predictions on inputs that are
  in-distribution but mislabeled in training data).
- This is a research/portfolio project, **not a validated medical device**.
  It is not intended for clinical use without regulatory clearance,
  extensive external validation, and clinician oversight.

---

## License

MIT — see `LICENSE`.
