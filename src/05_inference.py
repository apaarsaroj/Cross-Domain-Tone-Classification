"""
Script to run inference on Ground Truth dataset (Dataset A) using trained models.
Uses Receiver Model B and Receiver Model C to predict tone classifications.
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm
import numpy as np

print("=" * 60)
print("Inference on Ground Truth Dataset (Dataset A)")
print("=" * 60)

# Step 1: Load the 'test_a.csv' (Dataset A)
print("\n" + "=" * 60)
print("Step 1: Loading test_a.csv (Dataset A)")
print("=" * 60)

test_a_file = 'data/processed/test_a.csv'
if not os.path.exists(test_a_file):
    print(f"ERROR: Test file not found: {test_a_file}")
    exit(1)

df_test = pd.read_csv(test_a_file)
print(f"Loaded {len(df_test)} samples from {test_a_file}")
print(f"Columns: {df_test.columns.tolist()}")
print(f"\nLabel distribution:")
print(df_test['label'].value_counts().sort_index())

# Prepare the output dataframe
# According to requirements: original_text, true_label, pred_b, pred_c
df_results = pd.DataFrame()
df_results['original_text'] = df_test['text']
df_results['true_label'] = df_test['label']

# Step 2: Load 'models/receiver_b/' and run inference
print("\n" + "=" * 60)
print("Step 2: Loading Receiver Model B and running inference")
print("=" * 60)

model_b_dir = 'models/receiver_b'
if not os.path.exists(model_b_dir):
    print(f"ERROR: Model directory not found: {model_b_dir}")
    exit(1)

print(f"Loading model and tokenizer from: {model_b_dir}")
tokenizer_b = BertTokenizer.from_pretrained(model_b_dir)
model_b = BertForSequenceClassification.from_pretrained(model_b_dir)
model_b.eval()  # Set to evaluation mode
print("Model B loaded successfully!")

# Run inference
print(f"\nRunning inference on {len(df_test)} samples...")
predictions_b = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_b.to(device)
print(f"Using device: {device}")

batch_size = 32  # Process in batches for efficiency

with torch.no_grad():
    for i in tqdm(range(0, len(df_test), batch_size), desc="Processing batches"):
        batch_texts = df_test['text'].iloc[i:i+batch_size].tolist()
        
        # Tokenize
        encodings = tokenizer_b(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Get predictions
        outputs = model_b(**encodings)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        predictions_b.extend(batch_predictions)

df_results['pred_b'] = predictions_b
print(f"\nInference completed! Predictions saved to 'pred_b' column.")
print(f"Prediction distribution (Model B):")
print(pd.Series(predictions_b).value_counts().sort_index())

# Step 3: Load 'models/receiver_c/' and run inference (if available)
print("\n" + "=" * 60)
print("Step 3: Loading Receiver Model C and running inference")
print("=" * 60)

model_c_dir = 'models/receiver_c'
if os.path.exists(model_c_dir):
    # Check if model files exist
    model_files = ['config.json', 'model.safetensors']
    model_exists = all(os.path.exists(os.path.join(model_c_dir, f)) for f in model_files)
    
    if model_exists:
        print(f"Loading model and tokenizer from: {model_c_dir}")
        tokenizer_c = BertTokenizer.from_pretrained(model_c_dir)
        model_c = BertForSequenceClassification.from_pretrained(model_c_dir)
        model_c.eval()
        print("Model C loaded successfully!")
        
        # Run inference
        print(f"\nRunning inference on {len(df_test)} samples...")
        predictions_c = []
        model_c.to(device)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(df_test), batch_size), desc="Processing batches"):
                batch_texts = df_test['text'].iloc[i:i+batch_size].tolist()
                
                # Tokenize
                encodings = tokenizer_c(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                # Move to device
                encodings = {k: v.to(device) for k, v in encodings.items()}
                
                # Get predictions
                outputs = model_c(**encodings)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                
                predictions_c.extend(batch_predictions)
        
        df_results['pred_c'] = predictions_c
        print(f"\nInference completed! Predictions saved to 'pred_c' column.")
        print(f"Prediction distribution (Model C):")
        print(pd.Series(predictions_c).value_counts().sort_index())
    else:
        print(f"WARNING: Model C directory exists but model files are not complete.")
        print("Skipping Model C inference. Will set pred_c to None.")
        df_results['pred_c'] = None
else:
    print(f"WARNING: Model C directory not found: {model_c_dir}")
    print("Model C is still training or not available.")
    print("Skipping Model C inference. Will set pred_c to None.")
    df_results['pred_c'] = None

# Step 4: Save the resulting dataframe
print("\n" + "=" * 60)
print("Step 4: Saving results")
print("=" * 60)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, 'dataset_a_with_predictions.csv')

df_results.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
print(f"Total rows: {len(df_results)}")
print(f"\nColumns in output file:")
print(df_results.columns.tolist())
print(f"\nFirst few rows:")
print(df_results.head(10))

print("\n" + "=" * 60)
print("Inference completed successfully!")
print("=" * 60)



