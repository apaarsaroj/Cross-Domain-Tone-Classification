"""
Script to train BERT model for sequence classification.
Creates a reusable training function for BERT-based tone classification.
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
    """
    PyTorch Dataset class for tone classification.
    
    Args:
        texts: List of text strings
        labels: List of integer labels (0=Polite, 1=Professional, 2=Casual)
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length for tokenization
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize the text
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


def train_bert_model(train_file, val_file, output_dir):
    """
    Train a BERT model for sequence classification.
    
    Args:
        train_file: Path to training CSV file (must have 'text' and 'label' columns)
        val_file: Path to validation CSV file (must have 'text' and 'label' columns)
        output_dir: Directory to save the trained model and tokenizer
    
    Returns:
        None (saves model and tokenizer to output_dir)
    """
    print("=" * 60)
    print("BERT Model Training")
    print("=" * 60)
    
    # Step 1: Load the processed CSVs
    print(f"\nLoading training data from: {train_file}")
    train_df = pd.read_csv(train_file)
    print(f"Training data shape: {train_df.shape}")
    print(f"Training data columns: {train_df.columns.tolist()}")
    print(f"Training label distribution:")
    print(train_df['label'].value_counts().sort_index())
    
    print(f"\nLoading validation data from: {val_file}")
    val_df = pd.read_csv(val_file)
    print(f"Validation data shape: {val_df.shape}")
    print(f"Validation data columns: {val_df.columns.tolist()}")
    print(f"Validation label distribution:")
    print(val_df['label'].value_counts().sort_index())
    
    # Step 2: Initialize tokenizer
    print("\nInitializing BERT tokenizer (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Tokenizer loaded successfully!")
    
    # Step 3: Create PyTorch Dataset objects
    print("\nCreating PyTorch Dataset objects...")
    train_dataset = ToneDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    
    val_dataset = ToneDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Step 4: Initialize BertForSequenceClassification
    print("\nInitializing BertForSequenceClassification (num_labels=3)...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    print("Model initialized successfully!")
    
    # Step 5: Set up TrainingArguments
    print("\nSetting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',  # Changed from evaluation_strategy to eval_strategy for newer transformers
        save_strategy='epoch',
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        save_total_limit=3,
        seed=42
    )
    print("TrainingArguments configured:")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Evaluation strategy: {training_args.eval_strategy}")
    print(f"  - Save strategy: {training_args.save_strategy}")
    print(f"  - Output directory: {training_args.output_dir}")
    
    # Define compute_metrics function for evaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(labels, predictions)}
    
    # Step 6: Initialize Trainer
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
    
    # Step 7: Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Step 8: Save the final model and tokenizer
    print("\n" + "=" * 60)
    print("Saving model and tokenizer...")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel and tokenizer saved to: {output_dir}")
    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # This script is designed to be imported and called from other scripts
    # Do NOT run training here - just ensure the code is bug-free
    print("=" * 60)
    print("BERT Training Script")
    print("=" * 60)
    print("\nThis script defines the train_bert_model() function.")
    print("To train a model, import this function and call it with:")
    print("  from src.03_train_model import train_bert_model")
    print("  train_bert_model(train_file, val_file, output_dir)")
    print("\nCode is ready to be called. No training executed.")
    print("=" * 60)

