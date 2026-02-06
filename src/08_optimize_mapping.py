"""
Script to find optimal label mapping using calibration set.
Uses trained models B and C to predict on calibration set,
then finds the best mapping that maximizes F1 score.
"""

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import json
import os
from itertools import product
from tqdm import tqdm
import sys

print("=" * 60)
print("Label Mapping Optimization")
print("=" * 60)

# Load calibration set
print("\nLoading calibration set...")
cal_file = 'data/processed/calibration_a.csv'
df_cal = pd.read_csv(cal_file)
print(f"Calibration set: {len(df_cal)} samples")
print(f"True label distribution:")
print(df_cal['label'].value_counts().sort_index())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Model B and get predictions
print("\n" + "=" * 60)
print("Step 1: Getting predictions from Model B")
print("=" * 60)

model_b_dir = 'models/receiver_b_v2'
if not os.path.exists(model_b_dir):
    print(f"ERROR: Model B not found: {model_b_dir}")
    print("Please train Model B first using train_bert_model_v2")
    sys.exit(1)

print(f"Loading Model B from: {model_b_dir}")
tokenizer_b = BertTokenizer.from_pretrained(model_b_dir)
model_b = BertForSequenceClassification.from_pretrained(model_b_dir)
model_b.eval()
model_b.to(device)

# Load label mapping if exists
label_mapping_b_file = os.path.join(model_b_dir, 'label_mapping.json')
if os.path.exists(label_mapping_b_file):
    with open(label_mapping_b_file, 'r') as f:
        mapping_b = json.load(f)
    idx_to_label_b = {int(k): v for k, v in mapping_b['idx_to_label'].items()}
    print(f"Loaded label mapping for Model B: {len(idx_to_label_b)} labels")
else:
    # Assume labels are 0-indexed integers
    num_labels_b = model_b.config.num_labels
    idx_to_label_b = {i: i for i in range(num_labels_b)}
    print(f"Using default label mapping for Model B: {num_labels_b} labels")

# Get predictions from Model B
print("Running inference on calibration set...")
predictions_b_raw = []
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(df_cal), batch_size), desc="Model B inference"):
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
        predictions_b_raw.extend(batch_predictions)

print(f"Model B predictions: {len(predictions_b_raw)} samples")
print(f"Unique predicted labels: {sorted(set(predictions_b_raw))}")

# Load Model C and get predictions
print("\n" + "=" * 60)
print("Step 2: Getting predictions from Model C")
print("=" * 60)

model_c_dir = 'models/receiver_c_v2'
if not os.path.exists(model_c_dir):
    print(f"ERROR: Model C not found: {model_c_dir}")
    print("Please train Model C first using train_bert_model_v2")
    sys.exit(1)

print(f"Loading Model C from: {model_c_dir}")
tokenizer_c = BertTokenizer.from_pretrained(model_c_dir)
model_c = BertForSequenceClassification.from_pretrained(model_c_dir)
model_c.eval()
model_c.to(device)

# Load label mapping if exists
label_mapping_c_file = os.path.join(model_c_dir, 'label_mapping.json')
if os.path.exists(label_mapping_c_file):
    with open(label_mapping_c_file, 'r') as f:
        mapping_c = json.load(f)
    idx_to_label_c = {int(k): v for k, v in mapping_c['idx_to_label'].items()}
    print(f"Loaded label mapping for Model C: {len(idx_to_label_c)} labels")
else:
    num_labels_c = model_c.config.num_labels
    idx_to_label_c = {i: i for i in range(num_labels_c)}
    print(f"Using default label mapping for Model C: {num_labels_c} labels")

# Get predictions from Model C
print("Running inference on calibration set...")
predictions_c_raw = []

with torch.no_grad():
    for i in tqdm(range(0, len(df_cal), batch_size), desc="Model C inference"):
        batch_texts = df_cal['text'].iloc[i:i+batch_size].tolist()
        encodings = tokenizer_c(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        outputs = model_c(**encodings)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions_c_raw.extend(batch_predictions)

print(f"Model C predictions: {len(predictions_c_raw)} samples")
print(f"Unique predicted labels: {sorted(set(predictions_c_raw))}")

# Step 3: Find optimal mapping
print("\n" + "=" * 60)
print("Step 3: Finding Optimal Label Mapping")
print("=" * 60)

true_labels = df_cal['label'].values
target_labels = [0, 1, 2]  # Polite, Professional, Casual

# Get unique predicted labels
unique_b = sorted(set(predictions_b_raw))
unique_c = sorted(set(predictions_c_raw))

print(f"Model B unique predicted labels: {len(unique_b)}")
print(f"Model C unique predicted labels: {len(unique_c)}")

# For Model B: try all possible mappings
print("\nOptimizing mapping for Model B...")
print("This may take some time...")

best_f1_b = -1
best_mapping_b = None
best_predictions_b = None

# Use greedy approach: for each target label, assign the source label that gives best F1
# This is a simplified approach - for full optimization, we'd need to try all combinations
# which is computationally expensive (3^30 for B, 3^28 for C)

# Strategy: For each target label (0,1,2), find which source labels should map to it
# We'll use a greedy assignment based on majority voting

# For Model B
print("Finding optimal mapping for Model B using greedy assignment...")
source_labels_b = unique_b
target_labels_list = [0, 1, 2]

# Create a mapping matrix: for each source label, count how many times it appears with each target label
mapping_matrix_b = {}
for src_label in source_labels_b:
    mask = np.array(predictions_b_raw) == src_label
    if mask.sum() > 0:
        true_labels_for_src = true_labels[mask]
        counts = {0: np.sum(true_labels_for_src == 0),
                 1: np.sum(true_labels_for_src == 1),
                 2: np.sum(true_labels_for_src == 2)}
        mapping_matrix_b[src_label] = counts

# Greedy assignment: assign each source label to the target label it most often co-occurs with
mapping_b = {}
for src_label, counts in mapping_matrix_b.items():
    best_target = max(counts.items(), key=lambda x: x[1])[0]
    mapping_b[src_label] = best_target

# Apply mapping and evaluate
mapped_predictions_b = [mapping_b[pred] for pred in predictions_b_raw]
f1_b = f1_score(true_labels, mapped_predictions_b, average='macro')
print(f"Initial greedy mapping F1 (Model B): {f1_b:.4f}")

# Try to improve by swapping assignments
best_f1_b = f1_b
best_mapping_b = mapping_b.copy()
best_predictions_b = mapped_predictions_b.copy()

# Try swapping assignments for labels that are close
improved = True
iterations = 0
max_iterations = 100

while improved and iterations < max_iterations:
    improved = False
    iterations += 1
    
    for src_label in source_labels_b:
        for new_target in target_labels_list:
            if mapping_b[src_label] == new_target:
                continue
            
            # Try swapping
            old_target = mapping_b[src_label]
            mapping_b[src_label] = new_target
            
            # Evaluate
            mapped_predictions = [mapping_b[pred] for pred in predictions_b_raw]
            f1 = f1_score(true_labels, mapped_predictions, average='macro')
            
            if f1 > best_f1_b:
                best_f1_b = f1
                best_mapping_b = mapping_b.copy()
                best_predictions_b = mapped_predictions.copy()
                improved = True
            else:
                # Revert
                mapping_b[src_label] = old_target
    
    if iterations % 10 == 0:
        print(f"  Iteration {iterations}: Best F1 = {best_f1_b:.4f}")

mapping_b = best_mapping_b
print(f"\nBest mapping for Model B - F1 Score: {best_f1_b:.4f}")

# For Model C: same approach
print("\nFinding optimal mapping for Model C using greedy assignment...")
source_labels_c = unique_c
mapping_matrix_c = {}
for src_label in source_labels_c:
    mask = np.array(predictions_c_raw) == src_label
    if mask.sum() > 0:
        true_labels_for_src = true_labels[mask]
        counts = {0: np.sum(true_labels_for_src == 0),
                 1: np.sum(true_labels_for_src == 1),
                 2: np.sum(true_labels_for_src == 2)}
        mapping_matrix_c[src_label] = counts

mapping_c = {}
for src_label, counts in mapping_matrix_c.items():
    best_target = max(counts.items(), key=lambda x: x[1])[0]
    mapping_c[src_label] = best_target

mapped_predictions_c = [mapping_c[pred] for pred in predictions_c_raw]
f1_c = f1_score(true_labels, mapped_predictions_c, average='macro')
print(f"Initial greedy mapping F1 (Model C): {f1_c:.4f}")

best_f1_c = f1_c
best_mapping_c = mapping_c.copy()
best_predictions_c = mapped_predictions_c.copy()

improved = True
iterations = 0
while improved and iterations < max_iterations:
    improved = False
    iterations += 1
    
    for src_label in source_labels_c:
        for new_target in target_labels_list:
            if mapping_c[src_label] == new_target:
                continue
            
            old_target = mapping_c[src_label]
            mapping_c[src_label] = new_target
            
            mapped_predictions = [mapping_c[pred] for pred in predictions_c_raw]
            f1 = f1_score(true_labels, mapped_predictions, average='macro')
            
            if f1 > best_f1_c:
                best_f1_c = f1
                best_mapping_c = mapping_c.copy()
                best_predictions_c = mapped_predictions.copy()
                improved = True
            else:
                mapping_c[src_label] = old_target
    
    if iterations % 10 == 0:
        print(f"  Iteration {iterations}: Best F1 = {best_f1_c:.4f}")

mapping_c = best_mapping_c
print(f"\nBest mapping for Model C - F1 Score: {best_f1_c:.4f}")

# Save optimal mappings
print("\n" + "=" * 60)
print("Step 4: Saving Optimal Mappings")
print("=" * 60)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Convert label indices to label names if possible
mapping_b_named = {}
for idx, target in mapping_b.items():
    label_name = idx_to_label_b.get(idx, f"label_{idx}")
    mapping_b_named[label_name] = target

mapping_c_named = {}
for idx, target in mapping_c.items():
    label_name = idx_to_label_c.get(idx, f"label_{idx}")
    mapping_c_named[label_name] = target

# Save mappings
mapping_file_b = os.path.join(results_dir, 'optimal_mapping_b.json')
mapping_file_c = os.path.join(results_dir, 'optimal_mapping_c.json')

with open(mapping_file_b, 'w') as f:
    json.dump({
        'mapping': mapping_b,
        'mapping_named': mapping_b_named,
        'f1_score': float(best_f1_b)
    }, f, indent=2)

with open(mapping_file_c, 'w') as f:
    json.dump({
        'mapping': mapping_c,
        'mapping_named': mapping_c_named,
        'f1_score': float(best_f1_c)
    }, f, indent=2)

print(f"Saved optimal mapping for Model B: {mapping_file_b}")
print(f"Saved optimal mapping for Model C: {mapping_file_c}")

# Print mapping summary
print("\n" + "=" * 60)
print("Optimal Mapping Summary")
print("=" * 60)

print("\nModel B Mapping (F1 = {:.4f}):".format(best_f1_b))
for label_name, target in sorted(mapping_b_named.items()):
    target_name = ['Polite', 'Professional', 'Casual'][target]
    print(f"  {label_name} -> {target_name} ({target})")

print("\nModel C Mapping (F1 = {:.4f}):".format(best_f1_c))
for label_name, target in sorted(mapping_c_named.items()):
    target_name = ['Polite', 'Professional', 'Casual'][target]
    print(f"  {label_name} -> {target_name} ({target})")

# Evaluate on calibration set
print("\n" + "=" * 60)
print("Calibration Set Performance")
print("=" * 60)

print("\nModel B (with optimal mapping):")
print(classification_report(true_labels, best_predictions_b, 
                          target_names=['Polite', 'Professional', 'Casual'],
                          labels=[0, 1, 2]))

print("\nModel C (with optimal mapping):")
print(classification_report(true_labels, best_predictions_c,
                          target_names=['Polite', 'Professional', 'Casual'],
                          labels=[0, 1, 2]))

print("\n" + "=" * 60)
print("Mapping optimization completed!")
print("=" * 60)



