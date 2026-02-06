# Cross-Domain Tone Classification
A student project that explores how to **transfer models trained on different label systems** into a **shared, human‑meaningful target**: `Polite`, `Professional`, `Casual`.

Instead of guessing label mappings by hand, I learned the mapping from a small **calibration split** of the target dataset. This makes the system more honest about uncertainty and more defensible in real settings.

---

## Why This Matters

When datasets label text in different ways (tone vs. emotions), models can’t be compared or combined directly. I designed a pipeline that:

- keeps each dataset’s original labels intact,
- trains separate models,
- and **learns how to translate** each label system into the shared 3‑class target.

This mirrors a real‑world challenge: *systems often need to “talk” across incompatible label spaces.*

---

## What I Built

**Pipeline Overview**
1. **Preprocess** Dataset A (ground truth) into a long format, then split into calibration/test.
2. **Train** two BERT models on Dataset B (Tone Analysis) and Dataset C (GoEmotions), keeping their original labels.
3. **Learn an optimal mapping** from each model’s label space → {Polite, Professional, Casual} using the calibration set.
4. **Evaluate** on a held‑out test set and visualize errors.

---

## Key Results (Test Set, 4,800 samples)

| Model | Training Data | Accuracy | Macro F1 |
|------|---------------|----------|----------|
| Model B | Tone Analysis (2,684 train) | **54.58%** | **0.5337** |
| Model C | GoEmotions (8,044 train) | 46.50% | 0.4635 |

**Interesting finding:** The smaller dataset (Tone Analysis) outperformed the larger one because it was **more semantically aligned** to tone (domain fit mattered more than size).

---

## Visuals (in `results/`)

- Confusion matrices: `results/confusion_matrix_model_b_v2.png`, `results/confusion_matrix_model_c_v2.png`
- Training curves: `results/training_curves_receiver_b_v2.png`, `results/training_curves_receiver_c_v2.png`
- Label distributions and mappings: `results/label_distribution_model_b.png`, `results/optimal_mapping_model_b.png`, etc.

See `results/IMAGE_DESCRIPTIONS.md` for human‑readable explanations of each plot.

---

## Quickstart (Reproducible)

```bash
pip install -r requirements.txt

# 1) Preprocess
python src/02_preprocess_data_v2.py

# 2) Train
python src/04_run_training_v2.py

# 3) Evaluate (learn mapping + test)
python src/10_evaluate_model_b.py
python src/11_evaluate_model_c.py

# 4) Visualize
python src/12_generate_confusion_matrices.py
python src/13_visualize_training.py
python src/14_visualize_labels.py
```

---

## Project Structure

```
.
├── data/
│   └── processed/
├── models/
│   ├── receiver_b_v2/
│   └── receiver_c_v2/
├── results/
│   ├── *.png
│   ├── *.csv
│   └── final_report.md
├── src/
│   ├── 01_load_data.py
│   ├── 02_preprocess_data_v2.py
│   ├── 03_train_model_v2.py
│   ├── 04_run_training_v2.py
│   ├── 10_evaluate_model_b.py
│   ├── 11_evaluate_model_c.py
│   ├── 12_generate_confusion_matrices.py
│   ├── 13_visualize_training.py
│   └── 14_visualize_labels.py
└── requirements.txt
```

---

## What I Learned

- **Domain alignment beats dataset size** in transfer settings.
- A small, clean calibration set can make **cross‑label mapping tractable**.
- Visualization (confusion matrices + label distribution charts) surfaced where the models misunderstood tone.

---

## Limitations

- Label mapping is greedy and not guaranteed optimal.
- Dataset A is synthetic (rewrites of the same base sentences).
- This is a single‑language, single‑domain experiment.

---

## If I Had More Time

- Try optimal transport or Bayesian mapping for labels.
- Add a calibration set from a different domain to test robustness.
- Explore prompt‑tuning or adapter‑based transfer instead of full fine‑tuning.

---

## Portfolio Extras

- One‑page project site: `docs/index.html`
- Notebook walkthrough: `notebooks/portfolio_walkthrough.ipynb`
- Detailed technical report: `results/final_report.md`

---

If you’re a recruiter: I’m happy to walk through tradeoffs, decisions, and what I would improve next.
