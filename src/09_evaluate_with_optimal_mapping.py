"""
Script to evaluate models on test set using optimal label mappings.
"""

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import json
import os
from tqdm import tqdm

print("=" * 60)
print("Evaluation with Optimal Label Mapping")
print("=" * 60)

# Load test set
print("\nLoading test set...")
test_file = 'data/processed/test_a_v2.csv'
df_test = pd.read_csv(test_file)
print(f"Test set: {len(df_test)} samples")
print(f"True label distribution:")
print(df_test['label'].value_counts().sort_index())

# Load optimal mappings
print("\nLoading optimal mappings...")
mapping_file_b = 'results/optimal_mapping_b.json'
mapping_file_c = 'results/optimal_mapping_c.json'

if not os.path.exists(mapping_file_b) or not os.path.exists(mapping_file_c):
    print("ERROR: Optimal mappings not found!")
    print("Please run 08_optimize_mapping.py first")
    exit(1)

with open(mapping_file_b, 'r') as f:
    mapping_data_b = json.load(f)
    mapping_b = {int(k): v for k, v in mapping_data_b['mapping'].items()}

with open(mapping_file_c, 'r') as f:
    mapping_data_c = json.load(f)
    mapping_c = {int(k): v for k, v in mapping_data_c['mapping'].items()}

print(f"Loaded mapping for Model B (F1 on calibration: {mapping_data_b['f1_score']:.4f})")
print(f"Loaded mapping for Model C (F1 on calibration: {mapping_data_c['f1_score']:.4f})")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Model B and get predictions
print("\n" + "=" * 60)
print("Model B: Getting predictions on test set")
print("=" * 60)

model_b_dir = 'models/receiver_b_v2'
tokenizer_b = BertTokenizer.from_pretrained(model_b_dir)
model_b = BertForSequenceClassification.from_pretrained(model_b_dir)
model_b.eval()
model_b.to(device)

# Load label mapping
label_mapping_b_file = os.path.join(model_b_dir, 'label_mapping.json')
if os.path.exists(label_mapping_b_file):
    with open(label_mapping_b_file, 'r') as f:
        mapping_b_full = json.load(f)
    idx_to_label_b = {int(k): v for k, v in mapping_b_full['idx_to_label'].items()}
else:
    num_labels_b = model_b.config.num_labels
    idx_to_label_b = {i: i for i in range(num_labels_b)}

predictions_b_raw = []
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(df_test), batch_size), desc="Model B inference"):
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
        predictions_b_raw.extend(batch_predictions)

# Apply optimal mapping
predictions_b = [mapping_b.get(pred, 2) for pred in predictions_b_raw]  # Default to Casual if not in mapping
print(f"Applied optimal mapping to Model B predictions")

# Load Model C and get predictions
print("\n" + "=" * 60)
print("Model C: Getting predictions on test set")
print("=" * 60)

model_c_dir = 'models/receiver_c_v2'
tokenizer_c = BertTokenizer.from_pretrained(model_c_dir)
model_c = BertForSequenceClassification.from_pretrained(model_c_dir)
model_c.eval()
model_c.to(device)

label_mapping_c_file = os.path.join(model_c_dir, 'label_mapping.json')
if os.path.exists(label_mapping_c_file):
    with open(label_mapping_c_file, 'r') as f:
        mapping_c_full = json.load(f)
    idx_to_label_c = {int(k): v for k, v in mapping_c_full['idx_to_label'].items()}
else:
    num_labels_c = model_c.config.num_labels
    idx_to_label_c = {i: i for i in range(num_labels_c)}

predictions_c_raw = []

with torch.no_grad():
    for i in tqdm(range(0, len(df_test), batch_size), desc="Model C inference"):
        batch_texts = df_test['text'].iloc[i:i+batch_size].tolist()
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

# Apply optimal mapping
predictions_c = [mapping_c.get(pred, 2) for pred in predictions_c_raw]
print(f"Applied optimal mapping to Model C predictions")

# Evaluate
print("\n" + "=" * 60)
print("Final Evaluation Results")
print("=" * 60)

true_labels = df_test['label'].values
label_names = ['Polite', 'Professional', 'Casual']

# Model B
print("\n" + "=" * 60)
print("Model B Results")
print("=" * 60)

accuracy_b = accuracy_score(true_labels, predictions_b)
f1_macro_b = f1_score(true_labels, predictions_b, average='macro')
f1_weighted_b = f1_score(true_labels, predictions_b, average='weighted')

print(f"\nOverall Metrics:")
print(f"  Accuracy: {accuracy_b:.4f} ({accuracy_b*100:.2f}%)")
print(f"  Macro F1: {f1_macro_b:.4f}")
print(f"  Weighted F1: {f1_weighted_b:.4f}")

print(f"\nClassification Report:")
print(classification_report(true_labels, predictions_b,
                          target_names=label_names, labels=[0, 1, 2]))

cm_b = confusion_matrix(true_labels, predictions_b, labels=[0, 1, 2])
print(f"\nConfusion Matrix:")
print("                    Predicted:")
print("                    Polite  Professional  Casual")
print(f"True: Polite        {cm_b[0,0]:4d}      {cm_b[0,1]:4d}      {cm_b[0,2]:4d}")
print(f"      Professional  {cm_b[1,0]:4d}      {cm_b[1,1]:4d}      {cm_b[1,2]:4d}")
print(f"      Casual        {cm_b[2,0]:4d}      {cm_b[2,1]:4d}      {cm_b[2,2]:4d}")

# Model C
print("\n" + "=" * 60)
print("Model C Results")
print("=" * 60)

accuracy_c = accuracy_score(true_labels, predictions_c)
f1_macro_c = f1_score(true_labels, predictions_c, average='macro')
f1_weighted_c = f1_score(true_labels, predictions_c, average='weighted')

print(f"\nOverall Metrics:")
print(f"  Accuracy: {accuracy_c:.4f} ({accuracy_c*100:.2f}%)")
print(f"  Macro F1: {f1_macro_c:.4f}")
print(f"  Weighted F1: {f1_weighted_c:.4f}")

print(f"\nClassification Report:")
print(classification_report(true_labels, predictions_c,
                          target_names=label_names, labels=[0, 1, 2]))

cm_c = confusion_matrix(true_labels, predictions_c, labels=[0, 1, 2])
print(f"\nConfusion Matrix:")
print("                    Predicted:")
print("                    Polite  Professional  Casual")
print(f"True: Polite        {cm_c[0,0]:4d}      {cm_c[0,1]:4d}      {cm_c[0,2]:4d}")
print(f"      Professional  {cm_c[1,0]:4d}      {cm_c[1,1]:4d}      {cm_c[1,2]:4d}")
print(f"      Casual        {cm_c[2,0]:4d}      {cm_c[2,1]:4d}      {cm_c[2,2]:4d}")

# Save results
print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Save predictions
df_results = pd.DataFrame({
    'original_text': df_test['text'],
    'true_label': df_test['label'],
    'pred_b': predictions_b,
    'pred_c': predictions_c
})
results_file = os.path.join(results_dir, 'test_results_optimal_mapping.csv')
df_results.to_csv(results_file, index=False)
print(f"Saved predictions to: {results_file}")

# Save summary
summary = {
    'model_b': {
        'accuracy': float(accuracy_b),
        'macro_f1': float(f1_macro_b),
        'weighted_f1': float(f1_weighted_b),
        'calibration_f1': mapping_data_b['f1_score']
    },
    'model_c': {
        'accuracy': float(accuracy_c),
        'macro_f1': float(f1_macro_c),
        'weighted_f1': float(f1_weighted_c),
        'calibration_f1': mapping_data_c['f1_score']
    }
}

summary_file = os.path.join(results_dir, 'evaluation_summary_optimal.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary to: {summary_file}")

print("\n" + "=" * 60)
print("Evaluation completed!")
print("=" * 60)



