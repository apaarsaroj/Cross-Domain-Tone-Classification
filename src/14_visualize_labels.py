"""
Visualize label distributions and optimal mappings.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

def plot_label_distribution(df, title, output_file):
    """Plot label distribution bar chart."""
    label_counts = df['label'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(label_counts)), label_counts.values, color='steelblue', alpha=0.7)
    plt.xlabel('Label Index', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(label_counts)), label_counts.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_optimal_mapping(mapping_file, model_name, output_file):
    """Visualize optimal label mapping."""
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    mapping = mapping_data.get('mapping', {})
    mapping_named = mapping_data.get('mapping_named', {})
    
    # Create mapping visualization
    target_labels = ['Polite (0)', 'Professional (1)', 'Casual (2)']
    source_to_target = {}
    
    for src_label, target_idx in mapping.items():
        target_label = target_labels[target_idx]
        if target_label not in source_to_target:
            source_to_target[target_label] = []
        source_to_target[target_label].append(src_label)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    target_labels_short = ['Polite', 'Professional', 'Casual']
    counts = [len(source_to_target.get(label, [])) for label in target_labels]
    
    bars = ax.bar(target_labels_short, counts, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    ax.set_xlabel('Target Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Source Labels Mapped', fontsize=12, fontweight='bold')
    ax.set_title(f'Optimal Label Mapping: {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def main():
    """Generate label distribution and mapping visualizations."""
    print("=" * 60)
    print("Generating Label Visualizations")
    print("=" * 60)
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot label distributions for training sets
    datasets = [
        ('data/processed/train_b_v2.csv', 'Model B Training Set Label Distribution', 'label_distribution_model_b.png'),
        ('data/processed/train_c_v2.csv', 'Model C Training Set Label Distribution', 'label_distribution_model_c.png')
    ]
    
    for data_file, title, output_name in datasets:
        if os.path.exists(data_file):
            print(f"\nProcessing {data_file}...")
            df = pd.read_csv(data_file)
            output_file = os.path.join(results_dir, output_name)
            plot_label_distribution(df, title, output_file)
        else:
            print(f"  Warning: {data_file} not found, skipping...")
    
    # Plot optimal mappings
    mappings = [
        ('results/optimal_mapping_b_v2.json', 'Model B (Tone Analysis)', 'optimal_mapping_model_b.png'),
        ('results/optimal_mapping_c_v2.json', 'Model C (GoEmotions)', 'optimal_mapping_model_c.png')
    ]
    
    for mapping_file, model_name, output_name in mappings:
        if os.path.exists(mapping_file):
            print(f"\nProcessing {mapping_file}...")
            output_file = os.path.join(results_dir, output_name)
            plot_optimal_mapping(mapping_file, model_name, output_file)
        else:
            print(f"  Warning: {mapping_file} not found, skipping...")
    
    print("\n" + "=" * 60)
    print("Label visualizations generated successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()


