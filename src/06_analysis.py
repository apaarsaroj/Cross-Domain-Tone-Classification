"""
Script to analyze the "Misunderstanding Gap" using confusion matrices.
Quantifies how well Receiver Models understand Sender Intent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

print("=" * 60)
print("Analysis: Misunderstanding Gap")
print("=" * 60)

# Step 1: Load 'results/dataset_a_with_predictions.csv'
print("\n" + "=" * 60)
print("Step 1: Loading predictions")
print("=" * 60)

results_file = 'results/dataset_a_with_predictions.csv'
if not os.path.exists(results_file):
    print(f"ERROR: Results file not found: {results_file}")
    exit(1)

df = pd.read_csv(results_file)
print(f"Loaded {len(df)} samples from {results_file}")
print(f"Columns: {df.columns.tolist()}")

# Check which predictions are available
has_pred_b = 'pred_b' in df.columns and df['pred_b'].notna().any()
has_pred_c = 'pred_c' in df.columns and df['pred_c'].notna().any()

print(f"\nAvailable predictions:")
print(f"  - pred_b: {'Yes' if has_pred_b else 'No'}")
print(f"  - pred_c: {'Yes' if has_pred_c else 'No'}")

if not has_pred_b and not has_pred_c:
    print("ERROR: No predictions available!")
    exit(1)

# Filter out rows with missing predictions
if has_pred_b:
    df_b = df[df['pred_b'].notna()].copy()
    print(f"\nDataset for Model B analysis: {len(df_b)} samples")
else:
    df_b = None

if has_pred_c:
    df_c = df[df['pred_c'].notna()].copy()
    print(f"Dataset for Model C analysis: {len(df_c)} samples")
else:
    df_c = None
    print("Model C predictions not available (still training)")

# Label names for visualization
label_names = ['Polite', 'Professional', 'Casual']

# Step 2: Generate Confusion Matrices
print("\n" + "=" * 60)
print("Step 2: Generating Confusion Matrices")
print("=" * 60)

# Matrix 1: True Label vs. Pred B
if has_pred_b:
    print("\nGenerating confusion matrix for Model B...")
    cm_b = confusion_matrix(df_b['true_label'], df_b['pred_b'], labels=[0, 1, 2])
    print("Confusion Matrix (Model B):")
    print(cm_b)
    print("\nRow = True Label (Sender Intent)")
    print("Column = Predicted Label (Receiver B Perception)")

# Matrix 2: True Label vs. Pred C
if has_pred_c:
    print("\nGenerating confusion matrix for Model C...")
    cm_c = confusion_matrix(df_c['true_label'], df_c['pred_c'], labels=[0, 1, 2])
    print("Confusion Matrix (Model C):")
    print(cm_c)
    print("\nRow = True Label (Sender Intent)")
    print("Column = Predicted Label (Receiver C Perception)")

# Step 3: Visualize using Seaborn Heatmaps
print("\n" + "=" * 60)
print("Step 3: Visualizing Confusion Matrices")
print("=" * 60)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

# Plot confusion matrix for Model B
if has_pred_b:
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
    plt.xlabel('Receiver B Perception', fontsize=12, fontweight='bold')
    plt.ylabel('Sender Intent', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix: Sender Intent vs Receiver B Perception', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file_b = os.path.join(results_dir, 'confusion_matrix_model_b.png')
    plt.savefig(output_file_b, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file_b}")
    plt.close()

# Plot confusion matrix for Model C
if has_pred_c:
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
    plt.xlabel('Receiver C Perception', fontsize=12, fontweight='bold')
    plt.ylabel('Sender Intent', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix: Sender Intent vs Receiver C Perception', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file_c = os.path.join(results_dir, 'confusion_matrix_model_c.png')
    plt.savefig(output_file_c, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file_c}")
    plt.close()

# Step 4: Calculate Metrics
print("\n" + "=" * 60)
print("Step 4: Calculating Classification Metrics")
print("=" * 60)

# Classification Report for Model B
if has_pred_b:
    print("\n" + "=" * 60)
    print("Classification Report: Model B")
    print("=" * 60)
    report_b = classification_report(
        df_b['true_label'],
        df_b['pred_b'],
        target_names=label_names,
        labels=[0, 1, 2],
        output_dict=False
    )
    print(report_b)
    
    # Get detailed metrics
    report_b_dict = classification_report(
        df_b['true_label'],
        df_b['pred_b'],
        target_names=label_names,
        labels=[0, 1, 2],
        output_dict=True
    )
    
    print("\nOverall Accuracy (Model B):")
    accuracy_b = (df_b['true_label'] == df_b['pred_b']).mean()
    print(f"  {accuracy_b:.4f} ({accuracy_b*100:.2f}%)")
    
    # Identify most common errors
    print("\n" + "-" * 60)
    print("Error Analysis: Model B")
    print("-" * 60)
    
    # Create error dataframe
    df_b['error'] = df_b['true_label'] != df_b['pred_b']
    errors_b = df_b[df_b['error']].copy()
    
    if len(errors_b) > 0:
        # Count error types
        error_counts = {}
        for idx, row in errors_b.iterrows():
            true_label = int(row['true_label'])
            pred_label = int(row['pred_b'])
            error_key = f"Sender meant {label_names[true_label]}, but Receiver B understood {label_names[pred_label]}"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        # Sort by count
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTotal errors: {len(errors_b)} ({len(errors_b)/len(df_b)*100:.2f}%)")
        print("\nMost common errors:")
        for i, (error_type, count) in enumerate(sorted_errors[:5], 1):
            print(f"  {i}. {error_type}: {count} occurrences ({count/len(errors_b)*100:.2f}% of errors)")
    else:
        print("No errors found!")

# Classification Report for Model C
if has_pred_c:
    print("\n" + "=" * 60)
    print("Classification Report: Model C")
    print("=" * 60)
    report_c = classification_report(
        df_c['true_label'],
        df_c['pred_c'],
        target_names=label_names,
        labels=[0, 1, 2],
        output_dict=False
    )
    print(report_c)
    
    print("\nOverall Accuracy (Model C):")
    accuracy_c = (df_c['true_label'] == df_c['pred_c']).mean()
    print(f"  {accuracy_c:.4f} ({accuracy_c*100:.2f}%)")
    
    # Identify most common errors
    print("\n" + "-" * 60)
    print("Error Analysis: Model C")
    print("-" * 60)
    
    df_c['error'] = df_c['true_label'] != df_c['pred_c']
    errors_c = df_c[df_c['error']].copy()
    
    if len(errors_c) > 0:
        error_counts = {}
        for idx, row in errors_c.iterrows():
            true_label = int(row['true_label'])
            pred_label = int(row['pred_c'])
            error_key = f"Sender meant {label_names[true_label]}, but Receiver C understood {label_names[pred_label]}"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTotal errors: {len(errors_c)} ({len(errors_c)/len(df_c)*100:.2f}%)")
        print("\nMost common errors:")
        for i, (error_type, count) in enumerate(sorted_errors[:5], 1):
            print(f"  {i}. {error_type}: {count} occurrences ({count/len(errors_c)*100:.2f}% of errors)")
    else:
        print("No errors found!")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if has_pred_b:
    print(f"\nModel B (Receiver B):")
    print(f"  - Accuracy: {accuracy_b:.4f} ({accuracy_b*100:.2f}%)")
    print(f"  - Total samples: {len(df_b)}")
    print(f"  - Correct predictions: {len(df_b) - len(errors_b) if has_pred_b and len(errors_b) > 0 else len(df_b[df_b['true_label'] == df_b['pred_b']])}")
    print(f"  - Incorrect predictions: {len(errors_b) if has_pred_b and len(errors_b) > 0 else 0}")

if has_pred_c:
    print(f"\nModel C (Receiver C):")
    print(f"  - Accuracy: {accuracy_c:.4f} ({accuracy_c*100:.2f}%)")
    print(f"  - Total samples: {len(df_c)}")
    print(f"  - Correct predictions: {len(df_c) - len(errors_c) if has_pred_c and len(errors_c) > 0 else len(df_c[df_c['true_label'] == df_c['pred_c']])}")
    print(f"  - Incorrect predictions: {len(errors_c) if has_pred_c and len(errors_c) > 0 else 0}")

if has_pred_b and has_pred_c:
    print(f"\nComparison:")
    if accuracy_b > accuracy_c:
        print(f"  - Model B performs better than Model C (difference: {accuracy_b - accuracy_c:.4f})")
    elif accuracy_c > accuracy_b:
        print(f"  - Model C performs better than Model B (difference: {accuracy_c - accuracy_b:.4f})")
    else:
        print(f"  - Both models perform equally")

print("\n" + "=" * 60)
print("Analysis completed successfully!")
print("=" * 60)



