"""
Generate confusion matrix visualizations for Model B and Model C.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

print("=" * 60)
print("Generating Confusion Matrices")
print("=" * 60)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

label_names = ['Polite', 'Professional', 'Casual']

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

# Load Model B results
print("\nLoading Model B test results...")
df_b = pd.read_csv('results/test_results_model_b_v2.csv')
print(f"Model B: {len(df_b)} samples")
print(f"Columns: {df_b.columns.tolist()}")

cm_b = confusion_matrix(df_b['true_label'], df_b['pred_b'], labels=[0, 1, 2])
print("\nConfusion Matrix (Model B):")
print(cm_b)

# Plot confusion matrix for Model B
print("\nCreating heatmap for Model B...")
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_b,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names,
    cbar_kws={'label': 'Count'}
)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix: Model B (Tone Analysis)', fontsize=14, fontweight='bold')
plt.tight_layout()

output_file_b = os.path.join(results_dir, 'confusion_matrix_model_b_v2.png')
plt.savefig(output_file_b, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file_b}")
plt.close()

# Load Model C results
print("\nLoading Model C test results...")
df_c = pd.read_csv('results/test_results_model_c_v2.csv')
print(f"Model C: {len(df_c)} samples")

cm_c = confusion_matrix(df_c['true_label'], df_c['pred_c'], labels=[0, 1, 2])
print("\nConfusion Matrix (Model C):")
print(cm_c)

# Plot confusion matrix for Model C
print("\nCreating heatmap for Model C...")
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_c,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=label_names,
    yticklabels=label_names,
    cbar_kws={'label': 'Count'}
)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix: Model C (GoEmotions)', fontsize=14, fontweight='bold')
plt.tight_layout()

output_file_c = os.path.join(results_dir, 'confusion_matrix_model_c_v2.png')
plt.savefig(output_file_c, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file_c}")
plt.close()

print("\n" + "=" * 60)
print("Confusion matrices generated successfully!")
print("=" * 60)

