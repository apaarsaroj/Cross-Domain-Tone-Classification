"""
Script to preprocess and standardize labels across all datasets.
Step 1: Define the Mapping Rules explicitly.
Step 2: Implement these mappings.
Step 3: Data Splitting.
Step 4: Save the processed dataframes.
"""

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Import the data loading logic from 01_load_data.py
# We'll load the data first
data_dir = 'data'
processed_dir = os.path.join(data_dir, 'processed')

print("=" * 60)
print("Step 1: Define the Mapping Rules")
print("=" * 60)

# ============================================================================
# Step 1: Define the Mapping Rules explicitly
# ============================================================================

# Target Labels: 0 = Polite, 1 = Professional, 2 = Casual
TARGET_LABELS = {
    0: 'Polite',
    1: 'Professional',
    2: 'Casual'
}

# Rule for Dataset B (Tone Analysis):
# Map to 'Polite' (0) if label is in:
DATASET_B_POLITE = ['appreciative', 'diplomatic', 'thoughtful', 'apologetic', 
                    'benevolent', 'admiring', 'altruistic']

# Map to 'Professional' (1) if label is in:
DATASET_B_PROFESSIONAL = ['informative', 'cautionary', 'direct', 'candid', 'assertive']

# Map to 'Casual' (2) for ALL other labels

def map_dataset_b(label):
    """
    Map Dataset B (Tone Analysis) labels to standardized categories.
    
    Args:
        label: Original label from Dataset B (case-insensitive)
    
    Returns:
        int: Mapped label (0=Polite, 1=Professional, 2=Casual)
    """
    # Convert to lowercase for case-insensitive matching
    label_lower = str(label).lower().strip()
    
    if label_lower in [l.lower() for l in DATASET_B_POLITE]:
        return 0  # Polite
    elif label_lower in [l.lower() for l in DATASET_B_PROFESSIONAL]:
        return 1  # Professional
    else:
        return 2  # Casual (all other labels)


# Rule for Dataset C (GoEmotions):
# Map to 'Polite' (0) if label is in:
DATASET_C_POLITE = ['admiration', 'caring', 'gratitude', 'love', 'optimism', 'relief']

# Map to 'Professional' (1) if label is in:
DATASET_C_PROFESSIONAL = ['approval', 'realization', 'neutral']

# Map to 'Casual' (2) for ALL other labels

def map_dataset_c(label):
    """
    Map Dataset C (GoEmotions) labels to standardized categories.
    
    Args:
        label: Original label from Dataset C (case-insensitive)
    
    Returns:
        int: Mapped label (0=Polite, 1=Professional, 2=Casual)
    """
    # Convert to lowercase for case-insensitive matching
    label_lower = str(label).lower().strip()
    
    if label_lower in [l.lower() for l in DATASET_C_POLITE]:
        return 0  # Polite
    elif label_lower in [l.lower() for l in DATASET_C_PROFESSIONAL]:
        return 1  # Professional
    else:
        return 2  # Casual (all other labels)


# Rule for Dataset A (Tone Adjustment):
# Ensure its existing labels (Polite, Professional, Casual) are mapped to 0, 1, 2 respectively

def map_dataset_a(label):
    """
    Map Dataset A (Tone Adjustment) labels to standardized categories.
    
    Args:
        label: Original label from Dataset A (should be 'Polite', 'Professional', or 'Casual')
    
    Returns:
        int: Mapped label (0=Polite, 1=Professional, 2=Casual)
    """
    # Convert to lowercase for case-insensitive matching
    label_lower = str(label).lower().strip()
    
    if label_lower == 'polite':
        return 0  # Polite
    elif label_lower == 'professional':
        return 1  # Professional
    elif label_lower == 'casual':
        return 2  # Casual
    else:
        # If label doesn't match expected values, raise an error
        raise ValueError(f"Unexpected label in Dataset A: {label}. Expected 'Polite', 'Professional', or 'Casual'.")


# Print mapping rules summary
print("\nTarget Labels:")
for key, value in TARGET_LABELS.items():
    print(f"  {key} = {value}")

print("\nDataset B (Tone Analysis) Mapping Rules:")
print(f"  -> Polite (0): {DATASET_B_POLITE}")
print(f"  -> Professional (1): {DATASET_B_PROFESSIONAL}")
print("  -> Casual (2): ALL other labels")

print("\nDataset C (GoEmotions) Mapping Rules:")
print(f"  -> Polite (0): {DATASET_C_POLITE}")
print(f"  -> Professional (1): {DATASET_C_PROFESSIONAL}")
print("  -> Casual (2): ALL other labels")

print("\nDataset A (Tone Adjustment) Mapping Rules:")
print("  -> Polite -> 0")
print("  -> Professional -> 1")
print("  -> Casual -> 2")

print("\n" + "=" * 60)
print("Step 1 completed: Mapping rules defined successfully!")
print("=" * 60)

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "=" * 60)
print("Loading Datasets")
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

# Standardize column names for Dataset A
if 'Original' in df_a.columns:
    df_a = df_a.rename(columns={'Original': 'text'})

print(f"Dataset A shape: {df_a.shape}")
print(f"Dataset A columns: {df_a.columns.tolist()}")

# Load Dataset B: tone_analysis.csv
print("\nLoading Dataset B: tone_analysis.csv (Training Data 1)")
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

# Load Dataset C: go_emotions.csv
print("\nLoading Dataset C: go_emotions.csv (Training Data 2)")
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

# Standardize column names for Dataset C
if 'text' not in df_c.columns:
    possible_text_cols = [col for col in df_c.columns if col.lower() in ['text', 'sentence', 'content', 'message']]
    if possible_text_cols:
        df_c = df_c.rename(columns={possible_text_cols[0]: 'text'})

# Extract label from emotion columns
emotion_cols = [col for col in df_c.columns if col not in ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 
                                                             'created_utc', 'rater_id', 'example_very_unclear']]
if emotion_cols:
    df_c['label'] = df_c[emotion_cols].idxmax(axis=1)
    df_c = df_c[['text', 'label']]

print(f"Dataset C shape: {df_c.shape}")
print(f"Dataset C columns: {df_c.columns.tolist()}")

# ============================================================================
# Step 2: Implement these mappings
# ============================================================================

print("\n" + "=" * 60)
print("Step 2: Implement these mappings")
print("=" * 60)

# Apply mapping to Dataset A
# Note: Dataset A has Original, Polite, Professional, Casual columns
# We need to use Original as text, but we need labels
# For now, we'll use the Original text and we'll need to determine labels later
# Actually, Dataset A might need special handling - let's check if there's a label column
# If not, we might need to use Original text without labels for now, or create labels

# For Dataset A, we'll use Original as text
# Since Dataset A is the ground truth, we might need to create a label column
# But based on the structure, it seems like each row has Original + 3 versions
# We'll use Original as text and need to determine the true label
# For now, let's assume we need to process Dataset A differently

# Actually, let me check: Dataset A might be used differently
# Since it's "The Ground Truth", maybe we need to use Original text
# and the labels might be determined from the context or from another source
# For now, I'll create a simplified version using Original text
# and we'll need to handle labels separately

# Apply mapping to Dataset A
# Dataset A is a parallel corpus with columns: Original, Polite, Professional, Casual
# We need to transform from "wide" format to "long" format
# Each row becomes 3 rows: one for each style (Polite=0, Professional=1, Casual=2)
# We ignore the Original column as its label is ambiguous

print("\nProcessing Dataset A...")
print("Transforming Dataset A from wide format to long format...")
print(f"Original Dataset A shape: {df_a.shape}")

# Create an empty list to store processed rows
processed_rows = []

# Iterate through each row of the raw dataframe
for idx, row in df_a.iterrows():
    # Extract text from 'Polite' column -> assign Label 0
    if 'Polite' in df_a.columns and pd.notna(row['Polite']):
        processed_rows.append({
            'text': row['Polite'],
            'label': 0  # Polite
        })
    
    # Extract text from 'Professional' column -> assign Label 1
    if 'Professional' in df_a.columns and pd.notna(row['Professional']):
        processed_rows.append({
            'text': row['Professional'],
            'label': 1  # Professional
        })
    
    # Extract text from 'Casual' column -> assign Label 2
    if 'Casual' in df_a.columns and pd.notna(row['Casual']):
        processed_rows.append({
            'text': row['Casual'],
            'label': 2  # Casual
        })

# Create a new DataFrame from these extracted rows
df_a_processed = pd.DataFrame(processed_rows)

print(f"Dataset A processed shape: {df_a_processed.shape}")
print(f"Expected shape: approximately {df_a.shape[0] * 3} rows (3x the original)")
print(f"Dataset A label distribution:")
print(df_a_processed['label'].value_counts().sort_index())

# Apply mapping to Dataset B
print("\nProcessing Dataset B...")
print(f"Original label distribution in Dataset B:")
print(df_b['label'].value_counts().head(10))
df_b_processed = df_b.copy()
df_b_processed['label'] = df_b_processed['label'].apply(map_dataset_b)
print(f"\nMapped label distribution in Dataset B:")
print(df_b_processed['label'].value_counts())
print(f"Dataset B processed shape: {df_b_processed.shape}")

# Apply mapping to Dataset C
print("\nProcessing Dataset C...")
print(f"Original label distribution in Dataset C (top 10):")
print(df_c['label'].value_counts().head(10))
df_c_processed = df_c.copy()
df_c_processed['label'] = df_c_processed['label'].apply(map_dataset_c)
print(f"\nMapped label distribution in Dataset C:")
print(df_c_processed['label'].value_counts())
print(f"Dataset C processed shape: {df_c_processed.shape}")

print("\n" + "=" * 60)
print("Step 2 completed: Mappings applied successfully!")
print("=" * 60)

# ============================================================================
# Step 3: Data Splitting
# ============================================================================

print("\n" + "=" * 60)
print("Step 3: Data Splitting")
print("=" * 60)

# Split Dataset B into Train (80%) and Validation (20%)
print("\nSplitting Dataset B...")
df_b_train, df_b_val = train_test_split(
    df_b_processed, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_b_processed['label']
)
print(f"Dataset B Train: {df_b_train.shape[0]} rows")
print(f"Dataset B Validation: {df_b_val.shape[0]} rows")
print(f"Dataset B Train label distribution:")
print(df_b_train['label'].value_counts().sort_index())
print(f"Dataset B Validation label distribution:")
print(df_b_val['label'].value_counts().sort_index())

# Split Dataset C into Train (80%) and Validation (20%)
print("\nSplitting Dataset C...")
df_c_train, df_c_val = train_test_split(
    df_c_processed, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_c_processed['label']
)
print(f"Dataset C Train: {df_c_train.shape[0]} rows")
print(f"Dataset C Validation: {df_c_val.shape[0]} rows")
print(f"Dataset C Train label distribution:")
print(df_c_train['label'].value_counts().sort_index())
print(f"Dataset C Validation label distribution:")
print(df_c_val['label'].value_counts().sort_index())

# Do NOT split Dataset A (it will be used purely for testing/inference)
print("\nDataset A: No splitting (used for testing/inference)")
print(f"Dataset A: {df_a_processed.shape[0]} rows")

print("\n" + "=" * 60)
print("Step 3 completed: Data splitting completed successfully!")
print("=" * 60)

# ============================================================================
# Step 4: Save the processed dataframes
# ============================================================================

print("\n" + "=" * 60)
print("Step 4: Save the processed dataframes")
print("=" * 60)

# Create processed directory if it doesn't exist
os.makedirs(processed_dir, exist_ok=True)

# Save Dataset B splits
print("\nSaving Dataset B splits...")
df_b_train.to_csv(os.path.join(processed_dir, 'train_b.csv'), index=False)
df_b_val.to_csv(os.path.join(processed_dir, 'val_b.csv'), index=False)
print(f"  -> train_b.csv: {df_b_train.shape[0]} rows")
print(f"  -> val_b.csv: {df_b_val.shape[0]} rows")

# Save Dataset C splits
print("\nSaving Dataset C splits...")
df_c_train.to_csv(os.path.join(processed_dir, 'train_c.csv'), index=False)
df_c_val.to_csv(os.path.join(processed_dir, 'val_c.csv'), index=False)
print(f"  -> train_c.csv: {df_c_train.shape[0]} rows")
print(f"  -> val_c.csv: {df_c_val.shape[0]} rows")

# Save Dataset A (test set)
print("\nSaving Dataset A (test set)...")
# For Dataset A, we need to handle the label issue
# Since Dataset A structure is Original + 3 versions, we'll use Original as text
# For now, we'll save it without labels, or we need to determine labels
df_a_processed.to_csv(os.path.join(processed_dir, 'test_a.csv'), index=False)
print(f"  -> test_a.csv: {df_a_processed.shape[0]} rows")

print("\n" + "=" * 60)
print("Step 4 completed: All processed data saved successfully!")
print("=" * 60)

print("\n" + "=" * 60)
print("All steps completed successfully!")
print("=" * 60)
print(f"\nSummary:")
print(f"  Dataset A (test): {df_a_processed.shape[0]} rows -> test_a.csv")
print(f"  Dataset B Train: {df_b_train.shape[0]} rows -> train_b.csv")
print(f"  Dataset B Validation: {df_b_val.shape[0]} rows -> val_b.csv")
print(f"  Dataset C Train: {df_c_train.shape[0]} rows -> train_c.csv")
print(f"  Dataset C Validation: {df_c_val.shape[0]} rows -> val_c.csv")

