"""
Script to train two separate Receiver Models (B and C).
Imports the train_bert_model function from 03_train_model.py and trains both models.
"""

import os
import sys
import importlib.util

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import train_bert_model function from 03_train_model.py
# Note: Since module name starts with a number, we use importlib
train_model_path = os.path.join(os.path.dirname(__file__), '03_train_model.py')
spec = importlib.util.spec_from_file_location("train_model", train_model_path)
train_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_model_module)
train_bert_model = train_model_module.train_bert_model

print("=" * 60)
print("Training Receiver Models")
print("=" * 60)
print("\nThis script will train two separate BERT models:")
print("  - Receiver Model B (trained on Dataset B)")
print("  - Receiver Model C (trained on Dataset C)")
print("\nNote: Training may take a significant amount of time.")
print("=" * 60)

# Step 1: Train Receiver Model B
print("\n" + "=" * 60)
print("Step 1: Training Receiver Model B")
print("=" * 60)

train_b_file = 'data/processed/train_b.csv'
val_b_file = 'data/processed/val_b.csv'
output_b_dir = 'models/receiver_b'

# Check if files exist
if not os.path.exists(train_b_file):
    print(f"ERROR: Training file not found: {train_b_file}")
    sys.exit(1)
if not os.path.exists(val_b_file):
    print(f"ERROR: Validation file not found: {val_b_file}")
    sys.exit(1)

print(f"\nTraining files:")
print(f"  - Training: {train_b_file}")
print(f"  - Validation: {val_b_file}")
print(f"  - Output directory: {output_b_dir}")

# Create output directory if it doesn't exist
os.makedirs(output_b_dir, exist_ok=True)

# Train the model
try:
    train_bert_model(
        train_file=train_b_file,
        val_file=val_b_file,
        output_dir=output_b_dir
    )
    print("\n" + "=" * 60)
    print("Receiver Model B training completed successfully!")
    print("=" * 60)
except Exception as e:
    print(f"\nERROR: Failed to train Receiver Model B: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Train Receiver Model C
print("\n" + "=" * 60)
print("Step 2: Training Receiver Model C")
print("=" * 60)

train_c_file = 'data/processed/train_c.csv'
val_c_file = 'data/processed/val_c.csv'
output_c_dir = 'models/receiver_c'

# Check if files exist
if not os.path.exists(train_c_file):
    print(f"ERROR: Training file not found: {train_c_file}")
    sys.exit(1)
if not os.path.exists(val_c_file):
    print(f"ERROR: Validation file not found: {val_c_file}")
    sys.exit(1)

print(f"\nTraining files:")
print(f"  - Training: {train_c_file}")
print(f"  - Validation: {val_c_file}")
print(f"  - Output directory: {output_c_dir}")

# Create output directory if it doesn't exist
os.makedirs(output_c_dir, exist_ok=True)

# Train the model
try:
    train_bert_model(
        train_file=train_c_file,
        val_file=val_c_file,
        output_dir=output_c_dir
    )
    print("\n" + "=" * 60)
    print("Receiver Model C training completed successfully!")
    print("=" * 60)
except Exception as e:
    print(f"\nERROR: Failed to train Receiver Model C: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify models are saved
print("\n" + "=" * 60)
print("Verifying Model Files")
print("=" * 60)

model_b_files = [
    os.path.join(output_b_dir, 'config.json'),
    os.path.join(output_b_dir, 'pytorch_model.bin'),
    os.path.join(output_b_dir, 'tokenizer_config.json'),
    os.path.join(output_b_dir, 'vocab.txt')
]

model_c_files = [
    os.path.join(output_c_dir, 'config.json'),
    os.path.join(output_c_dir, 'pytorch_model.bin'),
    os.path.join(output_c_dir, 'tokenizer_config.json'),
    os.path.join(output_c_dir, 'vocab.txt')
]

print("\nReceiver Model B files:")
all_b_exist = True
for file in model_b_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {os.path.basename(file)}")
    if not exists:
        all_b_exist = False

print("\nReceiver Model C files:")
all_c_exist = True
for file in model_c_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {os.path.basename(file)}")
    if not exists:
        all_c_exist = False

if all_b_exist and all_c_exist:
    print("\n" + "=" * 60)
    print("All models saved successfully!")
    print("=" * 60)
    print(f"\nReceiver Model B: {output_b_dir}")
    print(f"Receiver Model C: {output_c_dir}")
else:
    print("\n" + "=" * 60)
    print("WARNING: Some model files are missing!")
    print("=" * 60)
    sys.exit(1)

print("\n" + "=" * 60)
print("Training Pipeline Completed Successfully!")
print("=" * 60)



