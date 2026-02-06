"""
Generate training curves from trainer_state.json files.
Visualizes loss, accuracy, and learning rate over training epochs.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_training_history(model_dir):
    """Load training history from trainer_state.json."""
    checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        return None
    
    # Find the latest checkpoint
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    trainer_state_file = os.path.join(model_dir, latest_checkpoint, 'trainer_state.json')
    
    if not os.path.exists(trainer_state_file):
        return None
    
    with open(trainer_state_file, 'r') as f:
        state = json.load(f)
    
    return state.get('log_history', [])

def extract_metrics(history):
    """Extract training metrics from history."""
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    learning_rate = []
    steps = []
    
    for entry in history:
        if 'step' in entry:
            steps.append(entry['step'])
        if 'loss' in entry and 'eval_loss' not in entry:
            train_loss.append((entry.get('step', len(train_loss) * 100), entry['loss']))
        if 'eval_loss' in entry:
            eval_loss.append((entry['step'], entry['eval_loss']))
        if 'eval_accuracy' in entry:
            eval_accuracy.append((entry['step'], entry['eval_accuracy']))
        if 'learning_rate' in entry:
            learning_rate.append((entry.get('step', len(learning_rate) * 100), entry['learning_rate']))
    
    return {
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'learning_rate': learning_rate,
        'steps': steps
    }

def plot_training_curves(metrics, model_name, output_file):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Curves: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    if metrics['train_loss']:
        steps, losses = zip(*metrics['train_loss'])
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_xlabel('Step', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Plot 2: Evaluation Loss
    if metrics['eval_loss']:
        steps, losses = zip(*metrics['eval_loss'])
        axes[0, 1].plot(steps, losses, 'r-', linewidth=2, marker='o', markersize=6, label='Validation Loss')
        axes[0, 1].set_xlabel('Step', fontsize=11)
        axes[0, 1].set_ylabel('Loss', fontsize=11)
        axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Plot 3: Evaluation Accuracy
    if metrics['eval_accuracy']:
        steps, accs = zip(*metrics['eval_accuracy'])
        axes[1, 0].plot(steps, accs, 'g-', linewidth=2, marker='s', markersize=6, label='Validation Accuracy')
        axes[1, 0].set_xlabel('Step', fontsize=11)
        axes[1, 0].set_ylabel('Accuracy', fontsize=11)
        axes[1, 0].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: Learning Rate
    if metrics['learning_rate']:
        steps, lrs = zip(*metrics['learning_rate'])
        axes[1, 1].plot(steps, lrs, 'm-', linewidth=2, label='Learning Rate')
        axes[1, 1].set_xlabel('Step', fontsize=11)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def main():
    """Generate training curves for both models."""
    print("=" * 60)
    print("Generating Training Curves")
    print("=" * 60)
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    models = [
        ('models/receiver_b_v2', 'Model B (Tone Analysis)'),
        ('models/receiver_c_v2', 'Model C (GoEmotions)')
    ]
    
    for model_dir, model_name in models:
        print(f"\nProcessing {model_name}...")
        
        if not os.path.exists(model_dir):
            print(f"  Warning: {model_dir} not found, skipping...")
            continue
        
        history = load_training_history(model_dir)
        if not history:
            print(f"  Warning: No training history found in {model_dir}")
            continue
        
        metrics = extract_metrics(history)
        output_file = os.path.join(results_dir, f'training_curves_{model_dir.split("/")[-1]}.png')
        plot_training_curves(metrics, model_name, output_file)
    
    print("\n" + "=" * 60)
    print("Training curves generated successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()


