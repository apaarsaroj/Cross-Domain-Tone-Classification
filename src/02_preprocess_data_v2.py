"""
Script to preprocess datasets WITHOUT pre-defined label mapping.
Dataset B and C keep their original labels for training.
Dataset A is split into calibration set (20%) and test set (80%).
"""

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

data_dir = 'data'
processed_dir = os.path.join(data_dir, 'processed')

print("=" * 60)
print("Preprocessing Data (Version 2: No Pre-defined Mapping)")
print("=" * 60)

# Load Dataset A: tone_adjustment.csv
print("\nLoading Dataset A: tone_adjustment.csv (The Ground Truth)")
dataset_a_path = None
if os.path.exists(os.path.join(data_dir, 'tone_adjustment.csv')):
    dataset_a_path = os.path.join(data_dir, 'tone_adjustment.csv')
elif os.path.exists(os.path.join(data_dir, 'datasetA', 'tone adjustment 1.csv')):
    dataset_a_path = os.path.join(data_dir, 'datasetA', 'tone adjustment 1.csv')
else:
    print("ERROR: Could not find tone_adjustment.csv")
    sys.exit(1)

encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
df_a = None
for encoding in encodings:
    try:
        df_a = pd.read_csv(dataset_a_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
if df_a is None:
    print(f"ERROR: Could not read {dataset_a_path} with any encoding")
    sys.exit(1)

if 'Original' in df_a.columns:
    df_a = df_a.rename(columns={'Original': 'text'})

print(f"Dataset A shape: {df_a.shape}")
print(f"Dataset A columns: {df_a.columns.tolist()}")

# Transform Dataset A from wide to long format
print("\nTransforming Dataset A from wide format to long format...")
processed_rows = []
for idx, row in df_a.iterrows():
    if 'Polite' in df_a.columns and pd.notna(row['Polite']):
        processed_rows.append({'text': row['Polite'], 'label': 0})  # Polite = 0
    if 'Professional' in df_a.columns and pd.notna(row['Professional']):
        processed_rows.append({'text': row['Professional'], 'label': 1})  # Professional = 1
    if 'Casual' in df_a.columns and pd.notna(row['Casual']):
        processed_rows.append({'text': row['Casual'], 'label': 2})  # Casual = 2

df_a_processed = pd.DataFrame(processed_rows)
print(f"Dataset A processed shape: {df_a_processed.shape}")

# Split Dataset A: 20% calibration, 80% test
print("\nSplitting Dataset A: 20% calibration set, 80% test set")
df_a_cal, df_a_test = train_test_split(
    df_a_processed,
    test_size=0.8,
    random_state=42,
    stratify=df_a_processed['label']
)
print(f"Calibration set: {len(df_a_cal)} rows")
print(f"Test set: {len(df_a_test)} rows")
print(f"Calibration label distribution:")
print(df_a_cal['label'].value_counts().sort_index())
print(f"Test label distribution:")
print(df_a_test['label'].value_counts().sort_index())

# Load Dataset B: tone_analysis.csv (KEEP ORIGINAL LABELS)
print("\nLoading Dataset B: tone_analysis.csv (Training Data 1 - Original Labels)")
dataset_b_path = None
if os.path.exists(os.path.join(data_dir, 'tone_analysis.csv')):
    dataset_b_path = os.path.join(data_dir, 'tone_analysis.csv')
elif os.path.exists(os.path.join(data_dir, 'datasetB', 'tone_v1.txt')):
    dataset_b_path = os.path.join(data_dir, 'datasetB', 'tone_v1.txt')
    data = []
    file_content = None
    for encoding in encodings:
        try:
            with open(dataset_b_path, 'r', encoding=encoding) as f:
                file_content = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    if file_content is None:
        print(f"ERROR: Could not read {dataset_b_path} with any encoding")
        sys.exit(1)
    for line in file_content:
        line = line.strip()
        if ' || ' in line:
            parts = line.split(' || ', 1)
            if len(parts) == 2:
                data.append({'text': parts[0], 'label': parts[1].rstrip('.')})
    df_b = pd.DataFrame(data)
else:
    print("ERROR: Could not find tone_analysis.csv")
    sys.exit(1)

print(f"Dataset B shape: {df_b.shape}")
print(f"Dataset B columns: {df_b.columns.tolist()}")
print(f"Dataset B unique labels: {sorted(df_b['label'].unique())}")
print(f"Dataset B label distribution:")
print(df_b['label'].value_counts().head(20))

# Split Dataset B into Train (80%) and Validation (20%)
print("\nSplitting Dataset B...")
df_b_train, df_b_val = train_test_split(
    df_b,
    test_size=0.2,
    random_state=42,
    stratify=df_b['label']
)
print(f"Dataset B Train: {df_b_train.shape[0]} rows")
print(f"Dataset B Validation: {df_b_val.shape[0]} rows")

# Load Dataset C: go_emotions.csv (KEEP ORIGINAL LABELS)
print("\nLoading Dataset C: go_emotions.csv (Training Data 2 - Original Labels)")
dataset_c_path = None
if os.path.exists(os.path.join(data_dir, 'go_emotions.csv')):
    dataset_c_path = os.path.join(data_dir, 'go_emotions.csv')
    df_c = None
    for encoding in encodings:
        try:
            df_c = pd.read_csv(dataset_c_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if df_c is None:
        print(f"ERROR: Could not read {dataset_c_path} with any encoding")
        sys.exit(1)
elif os.path.exists(os.path.join(data_dir, 'datasetC', 'goemotions_1.csv')):
    goemotions_files = [
        os.path.join(data_dir, 'datasetC', 'goemotions_1.csv'),
        os.path.join(data_dir, 'datasetC', 'goemotions_2.csv'),
        os.path.join(data_dir, 'datasetC', 'goemotions_3.csv')
    ]
    dfs = []
    for file in goemotions_files:
        if os.path.exists(file):
            df_temp = None
            for encoding in encodings:
                try:
                    df_temp = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df_temp is not None:
                dfs.append(df_temp)
    if dfs:
        df_c = pd.concat(dfs, ignore_index=True)
    else:
        print("ERROR: Could not find go_emotions.csv files")
        sys.exit(1)
else:
    print("ERROR: Could not find go_emotions.csv")
    sys.exit(1)

# Extract label from emotion columns
if 'text' not in df_c.columns:
    possible_text_cols = [col for col in df_c.columns if col.lower() in ['text', 'sentence', 'content', 'message']]
    if possible_text_cols:
        df_c = df_c.rename(columns={possible_text_cols[0]: 'text'})

emotion_cols = [col for col in df_c.columns if col not in ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 
                                                             'created_utc', 'rater_id', 'example_very_unclear']]
if emotion_cols:
    df_c['label'] = df_c[emotion_cols].idxmax(axis=1)
    df_c = df_c[['text', 'label']]

print(f"Dataset C original shape: {df_c.shape}")
print(f"Dataset C columns: {df_c.columns.tolist()}")
print(f"Dataset C unique labels: {sorted(df_c['label'].unique())}")
print(f"Dataset C label distribution (top 10):")
print(df_c['label'].value_counts().head(10))

# Sample Dataset C to be 3x the size of Dataset B training set
print("\nSampling Dataset C...")
target_train_size = len(df_b_train) * 3  # 3x Dataset B training size
target_total_size = int(target_train_size / 0.8)  # Account for 80/20 split

if len(df_c) > target_total_size:
    print(f"Sampling Dataset C from {len(df_c)} to {target_total_size} samples (3x Dataset B)")
    df_c_sampled = df_c.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(target_total_size * len(x) / len(df_c))), random_state=42)
    ).reset_index(drop=True)
    
    # If we need more samples, do additional stratified sampling
    if len(df_c_sampled) < target_total_size:
        remaining = target_total_size - len(df_c_sampled)
        df_c_remaining = df_c[~df_c.index.isin(df_c_sampled.index)]
        if len(df_c_remaining) > 0:
            df_c_additional = df_c_remaining.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(remaining * len(x) / len(df_c_remaining))), random_state=42)
            ).reset_index(drop=True)
            df_c_sampled = pd.concat([df_c_sampled, df_c_additional], ignore_index=True)
    
    df_c = df_c_sampled.sample(n=min(len(df_c_sampled), target_total_size), random_state=42).reset_index(drop=True)
    print(f"Sampled Dataset C shape: {df_c.shape}")
    print(f"Sampled Dataset C label distribution:")
    print(df_c['label'].value_counts().head(10))
else:
    print(f"Dataset C size ({len(df_c)}) is already smaller than target ({target_total_size}), keeping all samples")

# Split Dataset C into Train (80%) and Validation (20%)
print("\nSplitting Dataset C...")
df_c_train, df_c_val = train_test_split(
    df_c,
    test_size=0.2,
    random_state=42,
    stratify=df_c['label']
)
print(f"Dataset C Train: {df_c_train.shape[0]} rows")
print(f"Dataset C Validation: {df_c_val.shape[0]} rows")

# Save processed dataframes
print("\n" + "=" * 60)
print("Saving processed dataframes")
print("=" * 60)

os.makedirs(processed_dir, exist_ok=True)

# Save Dataset B splits (with original labels)
df_b_train.to_csv(os.path.join(processed_dir, 'train_b_v2.csv'), index=False)
df_b_val.to_csv(os.path.join(processed_dir, 'val_b_v2.csv'), index=False)
print(f"  -> train_b_v2.csv: {df_b_train.shape[0]} rows")
print(f"  -> val_b_v2.csv: {df_b_val.shape[0]} rows")

# Save Dataset C splits (with original labels)
df_c_train.to_csv(os.path.join(processed_dir, 'train_c_v2.csv'), index=False)
df_c_val.to_csv(os.path.join(processed_dir, 'val_c_v2.csv'), index=False)
print(f"  -> train_c_v2.csv: {df_c_train.shape[0]} rows")
print(f"  -> val_c_v2.csv: {df_c_val.shape[0]} rows")

# Save Dataset A splits
df_a_cal.to_csv(os.path.join(processed_dir, 'calibration_a.csv'), index=False)
df_a_test.to_csv(os.path.join(processed_dir, 'test_a_v2.csv'), index=False)
print(f"  -> calibration_a.csv: {len(df_a_cal)} rows")
print(f"  -> test_a_v2.csv: {len(df_a_test)} rows")

print("\n" + "=" * 60)
print("Preprocessing completed successfully!")
print("=" * 60)
print(f"\nSummary:")
print(f"  Dataset B Train: {df_b_train.shape[0]} rows (original labels)")
print(f"  Dataset B Validation: {df_b_val.shape[0]} rows (original labels)")
print(f"  Dataset C Train: {df_c_train.shape[0]} rows (original labels)")
print(f"  Dataset C Validation: {df_c_val.shape[0]} rows (original labels)")
print(f"  Dataset A Calibration: {len(df_a_cal)} rows")
print(f"  Dataset A Test: {len(df_a_test)} rows")

