"""
Script to train BERT model for sequence classification with original labels.
Automatically detects number of labels from training data.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score
import numpy as np
import os


class ToneDataset(Dataset):
    """PyTorch Dataset class for tone classification."""
    def __init__(self, texts, labels, tokenizer, max_length=128, label_to_idx=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert string labels to indices if needed
        if label_to_idx is not None:
            self.labels = [label_to_idx[label] for label in labels]
        else:
            # Assume labels are already integers
            self.labels = [int(label) for label in labels]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_bert_model_v2(train_file, val_file, output_dir):
    """
    Train a BERT model for sequence classification with original labels.
    
    Args:
        train_file: Path to training CSV file (must have 'text' and 'label' columns)
        val_file: Path to validation CSV file (must have 'text' and 'label' columns)
        output_dir: Directory to save the trained model and tokenizer
    """
    print("=" * 60)
    print("BERT Model Training (Version 2: Original Labels)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading training data from: {train_file}")
    train_df = pd.read_csv(train_file)
    print(f"Training data shape: {train_df.shape}")
    
    print(f"\nLoading validation data from: {val_file}")
    val_df = pd.read_csv(val_file)
    print(f"Validation data shape: {val_df.shape}")
    
    # Detect label type and create mapping
    train_labels = train_df['label'].unique()
    val_labels = val_df['label'].unique()
    all_labels = sorted(set(list(train_labels) + list(val_labels)))
    
    # Check if labels are strings or integers
    if isinstance(all_labels[0], str):
        # String labels - need to create mapping
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        num_labels = len(all_labels)
        print(f"\nDetected {num_labels} string labels")
        print(f"Label mapping created: {len(label_to_idx)} labels")
    else:
        # Integer labels
        num_labels = len(all_labels)
        label_to_idx = None
        idx_to_label = {idx: idx for idx in all_labels}
        print(f"\nDetected {num_labels} integer labels")
    
    print(f"Unique labels: {all_labels[:10]}{'...' if len(all_labels) > 10 else ''}")
    print(f"Label distribution (training):")
    print(train_df['label'].value_counts().head(10))
    
    # Initialize tokenizer
    print("\nInitializing BERT tokenizer (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Tokenizer loaded successfully!")
    
    # Create datasets
    print("\nCreating PyTorch Dataset objects...")
    train_dataset = ToneDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128,
        label_to_idx=label_to_idx
    )
    
    val_dataset = ToneDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128,
        label_to_idx=label_to_idx
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    print(f"\nInitializing BertForSequenceClassification (num_labels={num_labels})...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )
    print("Model initialized successfully!")
    
    # Define compute_metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(labels, predictions)}
    
    # Set up TrainingArguments
    print("\nSetting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        save_total_limit=3,
        seed=42
    )
    
    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Trainer initialized successfully!")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save model and tokenizer
    print("\n" + "=" * 60)
    print("Saving model and tokenizer...")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping if needed
    if label_to_idx is not None:
        import json
        mapping_file = os.path.join(output_dir, 'label_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump({'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}, f, indent=2)
        print(f"Label mapping saved to: {mapping_file}")
    
    print(f"\nModel and tokenizer saved to: {output_dir}")
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)



