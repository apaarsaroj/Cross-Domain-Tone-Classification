"""
Script to train two separate Receiver Models with original labels.
"""

import os
import sys
import importlib.util

# Import train_bert_model_v2 function
train_model_path = os.path.join(os.path.dirname(__file__), '03_train_model_v2.py')
spec = importlib.util.spec_from_file_location("train_model_v2", train_model_path)
train_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_model_module)
train_bert_model_v2 = train_model_module.train_bert_model_v2

print("=" * 60)
print("Training Receiver Models (Version 2: Original Labels)")
print("=" * 60)

# Step 1: Train Receiver Model B
print("\n" + "=" * 60)
print("Step 1: Training Receiver Model B")
print("=" * 60)

train_b_file = 'data/processed/train_b_v2.csv'
val_b_file = 'data/processed/val_b_v2.csv'
output_b_dir = 'models/receiver_b_v2'

if not os.path.exists(train_b_file):
    print(f"ERROR: Training file not found: {train_b_file}")
    sys.exit(1)
if not os.path.exists(val_b_file):
    print(f"ERROR: Validation file not found: {val_b_file}")
    sys.exit(1)

os.makedirs(output_b_dir, exist_ok=True)

try:
    train_bert_model_v2(
        train_file=train_b_file,
        val_file=val_b_file,
        output_dir=output_b_dir
    )
    print("\nReceiver Model B training completed!")
except Exception as e:
    print(f"\nERROR: Failed to train Receiver Model B: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Train Receiver Model C
print("\n" + "=" * 60)
print("Step 2: Training Receiver Model C")
print("=" * 60)

train_c_file = 'data/processed/train_c_v2.csv'
val_c_file = 'data/processed/val_c_v2.csv'
output_c_dir = 'models/receiver_c_v2'

if not os.path.exists(train_c_file):
    print(f"ERROR: Training file not found: {train_c_file}")
    sys.exit(1)
if not os.path.exists(val_c_file):
    print(f"ERROR: Validation file not found: {val_c_file}")
    sys.exit(1)

os.makedirs(output_c_dir, exist_ok=True)

try:
    train_bert_model_v2(
        train_file=train_c_file,
        val_file=val_c_file,
        output_dir=output_c_dir
    )
    print("\nReceiver Model C training completed!")
except Exception as e:
    print(f"\nERROR: Failed to train Receiver Model C: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All models trained successfully!")
print("=" * 60)



