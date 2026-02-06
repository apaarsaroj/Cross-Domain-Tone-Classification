"""
Script to load three datasets:
- Dataset A: tone_adjustment.csv (The Ground Truth)
- Dataset B: tone_analysis.csv (Training Data 1)
- Dataset C: go_emotions.csv (Training Data 2)
"""

import pandas as pd
import os

# Define file paths
data_dir = 'data'

# Try to load Dataset A: tone_adjustment.csv
print("=" * 60)
print("Loading Dataset A: tone_adjustment.csv (The Ground Truth)")
print("=" * 60)

# Check if file exists in data/ or subdirectories
dataset_a_path = None
if os.path.exists(os.path.join(data_dir, 'tone_adjustment.csv')):
    dataset_a_path = os.path.join(data_dir, 'tone_adjustment.csv')
elif os.path.exists(os.path.join(data_dir, 'datasetA', 'tone adjustment 1.csv')):
    dataset_a_path = os.path.join(data_dir, 'datasetA', 'tone adjustment 1.csv')
else:
    print("ERROR: Could not find tone_adjustment.csv")
    exit(1)

# Try different encodings
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
    exit(1)
print(f"\nDataset A shape: {df_a.shape}")
print("\nDataset A columns:", df_a.columns.tolist())
print("\nDataset A head:")
print(df_a.head())
print("\nDataset A info:")
print(df_a.info())

# Standardize column names for Dataset A
# Based on the structure, it has: Original, Polite, Professional, Casual
# We need to identify text and label columns
if 'Original' in df_a.columns:
    df_a = df_a.rename(columns={'Original': 'text'})
    # For ground truth, we might need to create a label column
    # This will be handled in preprocessing step
print("\nAfter column standardization:")
print("Dataset A columns:", df_a.columns.tolist())

# Try to load Dataset B: tone_analysis.csv
print("\n" + "=" * 60)
print("Loading Dataset B: tone_analysis.csv (Training Data 1)")
print("=" * 60)

dataset_b_path = None
if os.path.exists(os.path.join(data_dir, 'tone_analysis.csv')):
    dataset_b_path = os.path.join(data_dir, 'tone_analysis.csv')
elif os.path.exists(os.path.join(data_dir, 'datasetB', 'tone_v1.txt')):
    # Load from text file and convert to DataFrame
    dataset_b_path = os.path.join(data_dir, 'datasetB', 'tone_v1.txt')
    # Read as text file with || separator
    data = []
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
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
        exit(1)
    for line in file_content:
        line = line.strip()
        if ' || ' in line:
            parts = line.split(' || ', 1)
            if len(parts) == 2:
                data.append({'text': parts[0], 'label': parts[1].rstrip('.')})
    df_b = pd.DataFrame(data)
else:
    print("ERROR: Could not find tone_analysis.csv")
    exit(1)

print(f"\nDataset B shape: {df_b.shape}")
print("\nDataset B columns:", df_b.columns.tolist())
print("\nDataset B head:")
print(df_b.head())
print("\nDataset B info:")
print(df_b.info())

# Standardize column names for Dataset B
if 'label' not in df_b.columns:
    # Try to find label column
    possible_label_cols = [col for col in df_b.columns if col.lower() in ['label', 'tone', 'category', 'class']]
    if possible_label_cols:
        df_b = df_b.rename(columns={possible_label_cols[0]: 'label'})
if 'text' not in df_b.columns:
    possible_text_cols = [col for col in df_b.columns if col.lower() in ['text', 'sentence', 'content', 'message']]
    if possible_text_cols:
        df_b = df_b.rename(columns={possible_text_cols[0]: 'text'})

print("\nAfter column standardization:")
print("Dataset B columns:", df_b.columns.tolist())

# Try to load Dataset C: go_emotions.csv
print("\n" + "=" * 60)
print("Loading Dataset C: go_emotions.csv (Training Data 2)")
print("=" * 60)

dataset_c_path = None
if os.path.exists(os.path.join(data_dir, 'go_emotions.csv')):
    dataset_c_path = os.path.join(data_dir, 'go_emotions.csv')
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df_c = None
    for encoding in encodings:
        try:
            df_c = pd.read_csv(dataset_c_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if df_c is None:
        print(f"ERROR: Could not read {dataset_c_path} with any encoding")
        exit(1)
elif os.path.exists(os.path.join(data_dir, 'datasetC', 'goemotions_1.csv')):
    # Load and combine multiple goemotions files
    goemotions_files = [
        os.path.join(data_dir, 'datasetC', 'goemotions_1.csv'),
        os.path.join(data_dir, 'datasetC', 'goemotions_2.csv'),
        os.path.join(data_dir, 'datasetC', 'goemotions_3.csv')
    ]
    dfs = []
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
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
        exit(1)
else:
    print("ERROR: Could not find go_emotions.csv")
    exit(1)

print(f"\nDataset C shape: {df_c.shape}")
print("\nDataset C columns:", df_c.columns.tolist())
print("\nDataset C head:")
print(df_c.head())
print("\nDataset C info:")
print(df_c.info())

# Standardize column names for Dataset C
# GoEmotions has emotion columns (binary), we need to extract the label
# The label is the emotion column with value 1
if 'text' not in df_c.columns:
    possible_text_cols = [col for col in df_c.columns if col.lower() in ['text', 'sentence', 'content', 'message']]
    if possible_text_cols:
        df_c = df_c.rename(columns={possible_text_cols[0]: 'text'})

# Extract label from emotion columns (columns that are not metadata)
emotion_cols = [col for col in df_c.columns if col not in ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 
                                                             'created_utc', 'rater_id', 'example_very_unclear']]
if emotion_cols:
    # Create label column from the emotion with value 1
    df_c['label'] = df_c[emotion_cols].idxmax(axis=1)
    # Keep only text and label columns for now
    df_c = df_c[['text', 'label']]

print("\nAfter column standardization:")
print("Dataset C columns:", df_c.columns.tolist())
print("\nDataset C label distribution (top 10):")
print(df_c['label'].value_counts().head(10))

print("\n" + "=" * 60)
print("Data loading completed successfully!")
print("=" * 60)
print(f"\nSummary:")
print(f"Dataset A: {df_a.shape[0]} rows, {df_a.shape[1]} columns")
print(f"Dataset B: {df_b.shape[0]} rows, {df_b.shape[1]} columns")
print(f"Dataset C: {df_c.shape[0]} rows, {df_c.shape[1]} columns")

