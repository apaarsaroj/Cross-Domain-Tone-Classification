# Portfolio Summary: Cross-Domain Tone Classification

## Elevator Pitch (2–3 sentences)
I built a BERT-based system that learns how to **translate** different labeling schemes into one shared tone space: Polite, Professional, Casual. Instead of hand‑coding mappings, I used a calibration set to learn an optimal mapping that maximizes macro F1. This mirrors real‑world ML integration problems where label systems don’t match.

## What I Did (Recruiter‑Friendly)
- Trained two BERT classifiers on **Tone Analysis** and **GoEmotions** while preserving original label spaces.
- Designed a calibration‑driven label mapping step to align outputs to a shared 3‑class target.
- Evaluated with confusion matrices, macro F1, and accuracy; generated visual diagnostics.

## Results
- Model B (Tone Analysis): **54.58% accuracy**, **0.5337 macro F1**
- Model C (GoEmotions): **46.50% accuracy**, **0.4635 macro F1**
- Key insight: **domain alignment > dataset size** (Model B beat Model C despite 3x fewer samples).

## Skills Demonstrated
- NLP transfer learning and label space alignment
- Calibration‑based evaluation design
- Reproducible ML pipeline engineering
- Human‑readable reporting and visualization

## Links in This Repo
- Technical report: `results/final_report.md`
- Visuals: `results/`
- One‑page project site: `docs/index.html`
- Notebook walkthrough: `notebooks/portfolio_walkthrough.ipynb`
