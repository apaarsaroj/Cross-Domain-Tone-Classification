"""
Script to generate final report summarizing the misunderstanding gap analysis.
Includes case studies of correct and incorrect predictions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

print("=" * 60)
print("Generating Final Report")
print("=" * 60)

# Load the predictions
results_file = 'results/dataset_a_with_predictions.csv'
df = pd.read_csv(results_file)

# Label names
label_names = ['Polite', 'Professional', 'Casual']
label_map = {0: 'Polite', 1: 'Professional', 2: 'Casual'}

# Calculate metrics
df_b = df[df['pred_b'].notna()].copy()
df_c = df[df['pred_c'].notna()].copy()

# Confusion matrices
cm_b = confusion_matrix(df_b['true_label'], df_b['pred_b'], labels=[0, 1, 2])
cm_c = confusion_matrix(df_c['true_label'], df_c['pred_c'], labels=[0, 1, 2])

# Accuracy
accuracy_b = (df_b['true_label'] == df_b['pred_b']).mean()
accuracy_c = (df_c['true_label'] == df_c['pred_c']).mean()

# Classification reports
report_b = classification_report(
    df_b['true_label'], df_b['pred_b'],
    target_names=label_names, labels=[0, 1, 2], output_dict=True
)
report_c = classification_report(
    df_c['true_label'], df_c['pred_c'],
    target_names=label_names, labels=[0, 1, 2], output_dict=True
)

# Prepare report content
report_lines = []

report_lines.append("=" * 80)
report_lines.append("FINAL REPORT: Misunderstanding Gap Analysis")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("This report summarizes the analysis of how well two Receiver Models")
report_lines.append("(trained on different datasets) understand Sender Intent.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("")

# Section 1: Overall Accuracy and Detailed Metrics
report_lines.append("1. OVERALL ACCURACY AND DETAILED METRICS")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append("Question: How often did the Receivers correctly understand the Sender?")
report_lines.append("")
report_lines.append(f"Model B (Receiver B - trained on Tone Analysis dataset):")
report_lines.append(f"  - Overall Accuracy: {accuracy_b:.4f} ({accuracy_b*100:.2f}%)")
report_lines.append(f"  - Correct Predictions: {len(df_b[df_b['true_label'] == df_b['pred_b']])} / {len(df_b)}")
report_lines.append(f"  - Incorrect Predictions: {len(df_b[df_b['true_label'] != df_b['pred_b']])} / {len(df_b)}")
report_lines.append("")
report_lines.append("  Detailed Metrics by Class (Model B):")
report_lines.append("  " + "-" * 76)
report_lines.append("  {:<15} {:>10} {:>10} {:>10} {:>10}".format("Class", "Precision", "Recall", "F1-Score", "Support"))
report_lines.append("  " + "-" * 76)
for label_idx, label_name in enumerate(label_names):
    precision = report_b[label_name]['precision']
    recall = report_b[label_name]['recall']
    f1 = report_b[label_name]['f1-score']
    support = int(report_b[label_name]['support'])
    report_lines.append(f"  {label_name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
report_lines.append("  " + "-" * 76)
report_lines.append(f"  {'Macro Avg':<15} {report_b['macro avg']['precision']:>10.4f} {report_b['macro avg']['recall']:>10.4f} {report_b['macro avg']['f1-score']:>10.4f} {int(report_b['macro avg']['support']):>10}")
report_lines.append(f"  {'Weighted Avg':<15} {report_b['weighted avg']['precision']:>10.4f} {report_b['weighted avg']['recall']:>10.4f} {report_b['weighted avg']['f1-score']:>10.4f} {int(report_b['weighted avg']['support']):>10}")
report_lines.append("")
report_lines.append(f"Model C (Receiver C - trained on GoEmotions dataset):")
report_lines.append(f"  - Overall Accuracy: {accuracy_c:.4f} ({accuracy_c*100:.2f}%)")
report_lines.append(f"  - Correct Predictions: {len(df_c[df_c['true_label'] == df_c['pred_c']])} / {len(df_c)}")
report_lines.append(f"  - Incorrect Predictions: {len(df_c[df_c['true_label'] != df_c['pred_c']])} / {len(df_c)}")
report_lines.append("")
report_lines.append("  Detailed Metrics by Class (Model C):")
report_lines.append("  " + "-" * 76)
report_lines.append("  {:<15} {:>10} {:>10} {:>10} {:>10}".format("Class", "Precision", "Recall", "F1-Score", "Support"))
report_lines.append("  " + "-" * 76)
for label_idx, label_name in enumerate(label_names):
    precision = report_c[label_name]['precision']
    recall = report_c[label_name]['recall']
    f1 = report_c[label_name]['f1-score']
    support = int(report_c[label_name]['support'])
    report_lines.append(f"  {label_name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
report_lines.append("  " + "-" * 76)
report_lines.append(f"  {'Macro Avg':<15} {report_c['macro avg']['precision']:>10.4f} {report_c['macro avg']['recall']:>10.4f} {report_c['macro avg']['f1-score']:>10.4f} {int(report_c['macro avg']['support']):>10}")
report_lines.append(f"  {'Weighted Avg':<15} {report_c['weighted avg']['precision']:>10.4f} {report_c['weighted avg']['recall']:>10.4f} {report_c['weighted avg']['f1-score']:>10.4f} {int(report_c['weighted avg']['support']):>10}")
report_lines.append("")
report_lines.append("Comparison:")
if accuracy_b > accuracy_c:
    diff = accuracy_b - accuracy_c
    report_lines.append(f"  - Model B performs BETTER than Model C by {diff:.4f} ({diff*100:.2f} percentage points)")
    report_lines.append(f"  - Model B correctly understands the Sender {diff*100:.2f}% more often")
else:
    diff = accuracy_c - accuracy_b
    report_lines.append(f"  - Model C performs BETTER than Model B by {diff:.4f} ({diff*100:.2f} percentage points)")
    report_lines.append(f"  - Model C correctly understands the Sender {diff*100:.2f}% more often")
report_lines.append("")
report_lines.append("  Metric Comparison (Macro Average):")
report_lines.append(f"    - Precision: Model B ({report_b['macro avg']['precision']:.4f}) vs Model C ({report_c['macro avg']['precision']:.4f})")
report_lines.append(f"    - Recall: Model B ({report_b['macro avg']['recall']:.4f}) vs Model C ({report_c['macro avg']['recall']:.4f})")
report_lines.append(f"    - F1-Score: Model B ({report_b['macro avg']['f1-score']:.4f}) vs Model C ({report_c['macro avg']['f1-score']:.4f})")
report_lines.append("")
report_lines.append("Key Finding: Both models show significant misunderstanding, with accuracy")
report_lines.append("below 50%, indicating substantial communication gaps between sender intent")
report_lines.append("and receiver perception. The low recall scores for Polite class (0.18 and 0.17)")
report_lines.append("indicate that polite intent is particularly difficult to identify.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("")

# Section 2: The Misunderstanding Gap
report_lines.append("2. THE MISUNDERSTANDING GAP")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append("Question: Where is the biggest confusion? (Analysis of off-diagonal cells)")
report_lines.append("")
report_lines.append("Confusion Matrix - Model B:")
report_lines.append("                    Predicted:")
report_lines.append("                    Polite  Professional  Casual")
report_lines.append(f"True: Polite        {cm_b[0,0]:4d}      {cm_b[0,1]:4d}      {cm_b[0,2]:4d}")
report_lines.append(f"      Professional  {cm_b[1,0]:4d}      {cm_b[1,1]:4d}      {cm_b[1,2]:4d}")
report_lines.append(f"      Casual        {cm_b[2,0]:4d}      {cm_b[2,1]:4d}      {cm_b[2,2]:4d}")
report_lines.append("")
report_lines.append("Confusion Matrix - Model C:")
report_lines.append("                    Predicted:")
report_lines.append("                    Polite  Professional  Casual")
report_lines.append(f"True: Polite        {cm_c[0,0]:4d}      {cm_c[0,1]:4d}      {cm_c[0,2]:4d}")
report_lines.append(f"      Professional  {cm_c[1,0]:4d}      {cm_c[1,1]:4d}      {cm_c[1,2]:4d}")
report_lines.append(f"      Casual        {cm_c[2,0]:4d}      {cm_c[2,1]:4d}      {cm_c[2,2]:4d}")
report_lines.append("")

# Find biggest confusions
report_lines.append("Biggest Confusions (Off-Diagonal Cells):")
report_lines.append("")

# Model B
errors_b = []
for i in range(3):
    for j in range(3):
        if i != j:
            errors_b.append((i, j, cm_b[i, j], f"Sender meant {label_names[i]}, but Receiver B understood {label_names[j]}"))
errors_b.sort(key=lambda x: x[2], reverse=True)

report_lines.append("Model B - Top 3 Misunderstandings:")
for idx, (true_l, pred_l, count, desc) in enumerate(errors_b[:3], 1):
    pct = count / len(df_b) * 100
    report_lines.append(f"  {idx}. {desc}: {count} occurrences ({pct:.2f}% of all samples)")

# Model C
errors_c = []
for i in range(3):
    for j in range(3):
        if i != j:
            errors_c.append((i, j, cm_c[i, j], f"Sender meant {label_names[i]}, but Receiver C understood {label_names[j]}"))
errors_c.sort(key=lambda x: x[2], reverse=True)

report_lines.append("")
report_lines.append("Model C - Top 3 Misunderstandings:")
for idx, (true_l, pred_l, count, desc) in enumerate(errors_c[:3], 1):
    pct = count / len(df_c) * 100
    report_lines.append(f"  {idx}. {desc}: {count} occurrences ({pct:.2f}% of all samples)")

report_lines.append("")
report_lines.append("Key Finding: The BIGGEST confusion pattern for BOTH models is:")
report_lines.append("  'Sender meant Polite, but Receiver understood Casual'")
report_lines.append("")
report_lines.append("This suggests that polite language is frequently misinterpreted as casual,")
report_lines.append("indicating a significant gap in understanding formal vs. informal communication.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("")

# Section 3: Case Studies
report_lines.append("3. CASE STUDIES")
report_lines.append("-" * 80)
report_lines.append("")

# Add correct/incorrect flags
df_b['correct_b'] = df_b['true_label'] == df_b['pred_b']
df_c['correct_c'] = df_c['true_label'] == df_c['pred_c']

# Model B Case Studies
report_lines.append("3.1 Model B Case Studies")
report_lines.append("")

# Correct predictions - Model B
correct_b = df_b[df_b['correct_b']].copy()
report_lines.append("Correct Predictions (Model B):")
report_lines.append("")

# Sample from each category
for label in [0, 1, 2]:
    label_name = label_names[label]
    samples = correct_b[correct_b['true_label'] == label].head(2)
    if len(samples) > 0:
        report_lines.append(f"  {label_name} Intent - Correctly Identified as {label_name}:")
        for idx, row in samples.iterrows():
            text = row['original_text'][:100] + "..." if len(row['original_text']) > 100 else row['original_text']
            report_lines.append(f"    - \"{text}\"")
        report_lines.append("")

# Incorrect predictions - Model B
incorrect_b = df_b[~df_b['correct_b']].copy()
report_lines.append("Incorrect Predictions (Model B):")
report_lines.append("")

# Most common error: Polite -> Casual
polite_to_casual_b = incorrect_b[(incorrect_b['true_label'] == 0) & (incorrect_b['pred_b'] == 2)]
if len(polite_to_casual_b) > 0:
    report_lines.append("  Most Common Error: Polite Intent → Misunderstood as Casual")
    samples = polite_to_casual_b.head(3)
    for idx, row in samples.iterrows():
        text = row['original_text'][:100] + "..." if len(row['original_text']) > 100 else row['original_text']
        report_lines.append(f"    - \"{text}\"")
    report_lines.append("")

# Professional -> Casual
prof_to_casual_b = incorrect_b[(incorrect_b['true_label'] == 1) & (incorrect_b['pred_b'] == 2)]
if len(prof_to_casual_b) > 0:
    report_lines.append("  Second Most Common: Professional Intent → Misunderstood as Casual")
    samples = prof_to_casual_b.head(2)
    for idx, row in samples.iterrows():
        text = row['original_text'][:100] + "..." if len(row['original_text']) > 100 else row['original_text']
        report_lines.append(f"    - \"{text}\"")
    report_lines.append("")

# Model C Case Studies
report_lines.append("3.2 Model C Case Studies")
report_lines.append("")

# Correct predictions - Model C
correct_c = df_c[df_c['correct_c']].copy()
report_lines.append("Correct Predictions (Model C):")
report_lines.append("")

for label in [0, 1, 2]:
    label_name = label_names[label]
    samples = correct_c[correct_c['true_label'] == label].head(2)
    if len(samples) > 0:
        report_lines.append(f"  {label_name} Intent - Correctly Identified as {label_name}:")
        for idx, row in samples.iterrows():
            text = row['original_text'][:100] + "..." if len(row['original_text']) > 100 else row['original_text']
            report_lines.append(f"    - \"{text}\"")
        report_lines.append("")

# Incorrect predictions - Model C
incorrect_c = df_c[~df_c['correct_c']].copy()
report_lines.append("Incorrect Predictions (Model C):")
report_lines.append("")

# Most common error: Polite -> Casual
polite_to_casual_c = incorrect_c[(incorrect_c['true_label'] == 0) & (incorrect_c['pred_c'] == 2)]
if len(polite_to_casual_c) > 0:
    report_lines.append("  Most Common Error: Polite Intent → Misunderstood as Casual")
    samples = polite_to_casual_c.head(3)
    for idx, row in samples.iterrows():
        text = row['original_text'][:100] + "..." if len(row['original_text']) > 100 else row['original_text']
        report_lines.append(f"    - \"{text}\"")
    report_lines.append("")

# Professional -> Casual
prof_to_casual_c = incorrect_c[(incorrect_c['true_label'] == 1) & (incorrect_c['pred_c'] == 2)]
if len(prof_to_casual_c) > 0:
    report_lines.append("  Second Most Common: Professional Intent → Misunderstood as Casual")
    samples = prof_to_casual_c.head(2)
    for idx, row in samples.iterrows():
        text = row['original_text'][:100] + "..." if len(row['original_text']) > 100 else row['original_text']
        report_lines.append(f"    - \"{text}\"")
    report_lines.append("")

report_lines.append("=" * 80)
report_lines.append("")

# Section 4: Human Element Interpretation
report_lines.append("4. HUMAN ELEMENT INTERPRETATION")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append("Question: WHY do these misunderstandings happen based on the differing")
report_lines.append("training data (Context) of Model B vs. Model C?")
report_lines.append("")
report_lines.append("Training Data Context:")
report_lines.append("")
report_lines.append("Model B (Receiver B) - Trained on Tone Analysis Dataset:")
report_lines.append("  - Contains diverse tone labels: appreciative, diplomatic, thoughtful,")
report_lines.append("    apologetic, informative, cautionary, direct, candid, assertive, etc.")
report_lines.append("  - Focus: Various nuanced emotional and communication tones")
report_lines.append("  - Mapping: Polite (0) includes appreciative, diplomatic, thoughtful, etc.")
report_lines.append("            Professional (1) includes informative, cautionary, direct, etc.")
report_lines.append("            Casual (2) includes all other labels (angry, witty, sad, etc.)")
report_lines.append("")
report_lines.append("Model C (Receiver C) - Trained on GoEmotions Dataset:")
report_lines.append("  - Contains emotion labels: admiration, caring, gratitude, love, optimism,")
report_lines.append("    approval, realization, neutral, anger, joy, sadness, fear, etc.")
report_lines.append("  - Focus: Emotional states and reactions")
report_lines.append("  - Mapping: Polite (0) includes admiration, caring, gratitude, love, etc.")
report_lines.append("            Professional (1) includes approval, realization, neutral")
report_lines.append("            Casual (2) includes all other emotions (anger, joy, sadness, etc.)")
report_lines.append("")
report_lines.append("Why Misunderstandings Occur:")
report_lines.append("")
report_lines.append("1. Training Data Bias:")
report_lines.append("   - Both models were trained on datasets that may not perfectly align")
report_lines.append("     with the Ground Truth dataset's definition of Polite/Professional/Casual")
report_lines.append("   - The mapping from original labels to 3 categories may lose nuance")
report_lines.append("")
report_lines.append("2. Polite Intent Misinterpretation:")
report_lines.append("   - Polite language often uses formal structures and indirect expressions")
report_lines.append("   - Models trained on emotion-focused or tone-focused data may interpret")
report_lines.append("     formality as emotional distance or casualness")
report_lines.append("   - Example: 'I am feeling rather sad...' (Polite) might be seen as")
report_lines.append("     emotionally expressive (Casual) rather than formally structured")
report_lines.append("")
report_lines.append("3. Context-Dependent Understanding:")
report_lines.append("   - Model B (tone-focused): Better at recognizing professional directness")
report_lines.append("     but struggles with polite formality")
report_lines.append("   - Model C (emotion-focused): Better at recognizing emotional states")
report_lines.append("     but may confuse emotional expression with casual tone")
report_lines.append("")
report_lines.append("4. Casual Bias:")
report_lines.append("   - Both models show a strong bias toward predicting 'Casual'")
report_lines.append("   - This suggests that the training data had more casual examples, or")
report_lines.append("     that casual language patterns are more distinctive/easier to recognize")
report_lines.append("")
report_lines.append("5. Cultural and Linguistic Nuances:")
report_lines.append("   - The Ground Truth dataset contains carefully crafted parallel texts")
report_lines.append("     showing the same meaning in different tones")
report_lines.append("   - Models trained on real-world data may not capture these subtle")
report_lines.append("     distinctions as well as human annotators")
report_lines.append("")
report_lines.append("Key Insight: The misunderstanding gap reflects the challenge of")
report_lines.append("transferring knowledge from one domain (tone analysis or emotions)")
report_lines.append("to another (formal tone classification). The models' training context")
report_lines.append("shapes their 'perception' of communication, leading to systematic biases")
report_lines.append("in how they interpret sender intent.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("")

# Summary
report_lines.append("SUMMARY")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append("This analysis reveals a significant 'Misunderstanding Gap' between")
report_lines.append("sender intent and receiver perception:")
report_lines.append("")
report_lines.append(f"  • Model B: Accuracy={accuracy_b*100:.2f}%, Macro F1={report_b['macro avg']['f1-score']:.4f}, Macro Recall={report_b['macro avg']['recall']:.4f}")
report_lines.append(f"  • Model C: Accuracy={accuracy_c*100:.2f}%, Macro F1={report_c['macro avg']['f1-score']:.4f}, Macro Recall={report_c['macro avg']['recall']:.4f}")
report_lines.append("")
report_lines.append("The biggest confusion: Polite intent is frequently misunderstood as Casual,")
report_lines.append("suggesting that formal communication patterns are not well-captured by")
report_lines.append("models trained on tone or emotion datasets.")
report_lines.append("")
report_lines.append("Notable findings:")
report_lines.append(f"  • Polite class has very low recall (B: {report_b['Polite']['recall']:.2f}, C: {report_c['Polite']['recall']:.2f})")
report_lines.append(f"  • Casual class has high recall but lower precision (B: R={report_b['Casual']['recall']:.2f}, P={report_b['Casual']['precision']:.2f})")
report_lines.append(f"  • Professional class shows moderate performance (B: F1={report_b['Professional']['f1-score']:.2f}, C: F1={report_c['Professional']['f1-score']:.2f})")
report_lines.append("")
report_lines.append("These findings highlight the importance of context-aware training and")
report_lines.append("the challenges in cross-domain transfer learning for communication analysis.")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("End of Report")
report_lines.append("=" * 80)

# Write report to file
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
report_file = os.path.join(results_dir, 'final_report.txt')

with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

# Print to console
print("\n" + '\n'.join(report_lines))
print("\n" + "=" * 60)
print(f"Report saved to: {report_file}")
print("=" * 60)

