"""
Script to evaluate Model B:
1. On original validation set (with original labels)
2. On Dataset A test set (with optimal mapping from calibration set)
"""

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import json
import os
from tqdm import tqdm

print("=" * 60)
print("Model B Evaluation")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Model B
print("\n" + "=" * 60)
print("Loading Model B")
print("=" * 60)

model_b_dir = 'models/receiver_b_v2'
tokenizer_b = BertTokenizer.from_pretrained(model_b_dir)
model_b = BertForSequenceClassification.from_pretrained(model_b_dir)
model_b.eval()
model_b.to(device)

# Load label mapping
label_mapping_b_file = os.path.join(model_b_dir, 'label_mapping.json')
with open(label_mapping_b_file, 'r') as f:
    mapping_b_full = json.load(f)
label_to_idx_b = mapping_b_full['label_to_idx']
idx_to_label_b = {int(k): v for k, v in mapping_b_full['idx_to_label'].items()}

num_labels_b = len(label_to_idx_b)
print(f"Model B loaded: {num_labels_b} labels")
print(f"Label mapping: {len(label_to_idx_b)} original labels")

# Test 1: Evaluate on original validation set
print("\n" + "=" * 60)
print("Test 1: Evaluation on Original Validation Set")
print("=" * 60)

val_b_file = 'data/processed/val_b_v2.csv'
df_val_b = pd.read_csv(val_b_file)
print(f"Validation set: {len(df_val_b)} samples")
print(f"Original label distribution:")
print(df_val_b['label'].value_counts().head(10))

# Get predictions
print("\nRunning inference...")
predictions_b_raw = []
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(df_val_b), batch_size), desc="Inference"):
        batch_texts = df_val_b['text'].iloc[i:i+batch_size].tolist()
        encodings = tokenizer_b(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        outputs = model_b(**encodings)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions_b_raw.extend(batch_predictions)

# Convert predictions back to label names
predictions_b_labels = [idx_to_label_b[pred] for pred in predictions_b_raw]
true_labels = df_val_b['label'].values

# Calculate accuracy (exact match on original labels)
correct = sum(p == t for p, t in zip(predictions_b_labels, true_labels))
accuracy_original = correct / len(true_labels)

print(f"\nResults on Original Validation Set:")
print(f"  Accuracy: {accuracy_original:.4f} ({accuracy_original*100:.2f}%)")
print(f"  Correct: {correct} / {len(true_labels)}")

# Show some examples
print(f"\nSample predictions (first 10):")
for i in range(min(10, len(df_val_b))):
    print(f"  True: {true_labels[i]:<20} Pred: {predictions_b_labels[i]:<20} {'✓' if true_labels[i] == predictions_b_labels[i] else '✗'}")

# Test 2: Find optimal mapping on calibration set and evaluate on test set
print("\n" + "=" * 60)
print("Test 2: Optimal Mapping on Calibration Set")
print("=" * 60)

# Load calibration set
cal_file = 'data/processed/calibration_a.csv'
df_cal = pd.read_csv(cal_file)
print(f"Calibration set: {len(df_cal)} samples")
print(f"True label distribution:")
print(df_cal['label'].value_counts().sort_index())

# Get predictions on calibration set
print("\nRunning inference on calibration set...")
predictions_cal_raw = []

with torch.no_grad():
    for i in tqdm(range(0, len(df_cal), batch_size), desc="Calibration inference"):
        batch_texts = df_cal['text'].iloc[i:i+batch_size].tolist()
        encodings = tokenizer_b(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        outputs = model_b(**encodings)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions_cal_raw.extend(batch_predictions)

# Find optimal mapping
print("\nFinding optimal label mapping...")
true_labels_cal = df_cal['label'].values
target_labels = [0, 1, 2]  # Polite, Professional, Casual

# Get unique predicted labels
unique_predicted = sorted(set(predictions_cal_raw))

# Create mapping matrix: for each source label, count co-occurrence with target labels
mapping_matrix = {}
for src_idx in unique_predicted:
    mask = np.array(predictions_cal_raw) == src_idx
    if mask.sum() > 0:
        true_labels_for_src = true_labels_cal[mask]
        counts = {
            0: np.sum(true_labels_for_src == 0),
            1: np.sum(true_labels_for_src == 1),
            2: np.sum(true_labels_for_src == 2)
        }
        mapping_matrix[src_idx] = counts

# Greedy assignment
optimal_mapping = {}
for src_idx, counts in mapping_matrix.items():
    best_target = max(counts.items(), key=lambda x: x[1])[0]
    optimal_mapping[src_idx] = best_target

# Apply mapping and evaluate on calibration set
mapped_predictions_cal = [optimal_mapping[pred] for pred in predictions_cal_raw]
f1_cal = f1_score(true_labels_cal, mapped_predictions_cal, average='macro')
accuracy_cal = accuracy_score(true_labels_cal, mapped_predictions_cal)

print(f"\nOptimal Mapping Performance on Calibration Set:")
print(f"  Accuracy: {accuracy_cal:.4f} ({accuracy_cal*100:.2f}%)")
print(f"  Macro F1: {f1_cal:.4f}")

# Show mapping
print(f"\nOptimal Label Mapping (Model B):")
label_names = ['Polite', 'Professional', 'Casual']
for src_idx, target in sorted(optimal_mapping.items()):
    src_label = idx_to_label_b.get(src_idx, f"label_{src_idx}")
    target_name = label_names[target]
    count = mapping_matrix[src_idx][target]
    total = sum(mapping_matrix[src_idx].values())
    print(f"  {src_label:<20} -> {target_name} ({target}) [{count}/{total} samples]")

# Evaluate on test set
print("\n" + "=" * 60)
print("Test 2: Evaluation on Test Set (with Optimal Mapping)")
print("=" * 60)

test_file = 'data/processed/test_a_v2.csv'
df_test = pd.read_csv(test_file)
print(f"Test set: {len(df_test)} samples")
print(f"True label distribution:")
print(df_test['label'].value_counts().sort_index())

# Get predictions on test set
print("\nRunning inference on test set...")
predictions_test_raw = []

with torch.no_grad():
    for i in tqdm(range(0, len(df_test), batch_size), desc="Test inference"):
        batch_texts = df_test['text'].iloc[i:i+batch_size].tolist()
        encodings = tokenizer_b(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        outputs = model_b(**encodings)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions_test_raw.extend(batch_predictions)

# Apply optimal mapping
predictions_test = [optimal_mapping.get(pred, 2) for pred in predictions_test_raw]  # Default to Casual
true_labels_test = df_test['label'].values

# Calculate metrics
accuracy_test = accuracy_score(true_labels_test, predictions_test)
f1_macro_test = f1_score(true_labels_test, predictions_test, average='macro')
f1_weighted_test = f1_score(true_labels_test, predictions_test, average='weighted')

print(f"\nResults on Test Set (with Optimal Mapping):")
print(f"  Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"  Macro F1: {f1_macro_test:.4f}")
print(f"  Weighted F1: {f1_weighted_test:.4f}")

print(f"\nClassification Report:")
print(classification_report(true_labels_test, predictions_test,
                          target_names=label_names, labels=[0, 1, 2]))

cm_test = confusion_matrix(true_labels_test, predictions_test, labels=[0, 1, 2])
print(f"\nConfusion Matrix:")
print("                    Predicted:")
print("                    Polite  Professional  Casual")
print(f"True: Polite        {cm_test[0,0]:4d}      {cm_test[0,1]:4d}      {cm_test[0,2]:4d}")
print(f"      Professional  {cm_test[1,0]:4d}      {cm_test[1,1]:4d}      {cm_test[1,2]:4d}")
print(f"      Casual        {cm_test[2,0]:4d}      {cm_test[2,1]:4d}      {cm_test[2,2]:4d}")

# Save results
print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Save optimal mapping
mapping_file = os.path.join(results_dir, 'optimal_mapping_b_v2.json')
mapping_named = {idx_to_label_b.get(int(k), f"label_{k}"): int(v) for k, v in optimal_mapping.items()}
mapping_dict = {int(k): int(v) for k, v in optimal_mapping.items()}
with open(mapping_file, 'w') as f:
    json.dump({
        'mapping': mapping_dict,
        'mapping_named': mapping_named,
        'calibration_f1': float(f1_cal),
        'calibration_accuracy': float(accuracy_cal),
        'test_accuracy': float(accuracy_test),
        'test_f1_macro': float(f1_macro_test),
        'test_f1_weighted': float(f1_weighted_test)
    }, f, indent=2)
print(f"Saved optimal mapping to: {mapping_file}")

# Save test predictions
df_results = pd.DataFrame({
    'original_text': df_test['text'],
    'true_label': df_test['label'],
    'pred_b': predictions_test
})
results_file = os.path.join(results_dir, 'test_results_model_b_v2.csv')
df_results.to_csv(results_file, index=False)
print(f"Saved test predictions to: {results_file}")

print("\n" + "=" * 60)
print("Evaluation Summary")
print("=" * 60)
print(f"\n1. Original Validation Set (Original Labels):")
print(f"   Accuracy: {accuracy_original:.4f} ({accuracy_original*100:.2f}%)")
print(f"\n2. Test Set (with Optimal Mapping from Calibration Set):")
print(f"   Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"   Macro F1: {f1_macro_test:.4f}")
print(f"   Weighted F1: {f1_weighted_test:.4f}")
print(f"\n3. Calibration Set (used for mapping optimization):")
print(f"   Accuracy: {accuracy_cal:.4f} ({accuracy_cal*100:.2f}%)")
print(f"   Macro F1: {f1_cal:.4f}")
print("\n" + "=" * 60)

