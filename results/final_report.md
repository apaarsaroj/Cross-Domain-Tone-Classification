# Cross-Domain Tone Classification: A Data-Driven Label Mapping Approach

## Executive Summary

This report presents a cross-domain transfer learning approach for tone classification, where models trained on diverse label systems (Datasets B and C) are adapted to classify text into three target categories: Polite, Professional, and Casual (Dataset A). Instead of using pre-defined label mappings, we employ a data-driven approach that learns optimal mappings from a calibration set, maximizing F1-score performance.

**Key Results:**
- **Model B** (Tone Analysis): Test Accuracy 54.58%, Macro F1 0.5337
- **Model C** (GoEmotions): Test Accuracy 46.50%, Macro F1 0.4635
- Model B outperforms Model C by **8.08% accuracy** and **0.0702 F1-score** despite having **3× fewer training samples** (2,684 vs 8,044), primarily due to better domain alignment and more balanced label distribution.

---

## 1. Methodology Overview

### 1.1 Approach

Our methodology consists of four main stages:

1. **Data Preprocessing**: Load and preprocess three datasets, preserving original label systems for Datasets B and C
2. **Model Training**: Train separate BERT models on Datasets B and C using their original label systems
3. **Label Mapping Optimization**: Use a calibration set (20% of Dataset A) to find optimal mappings from original labels to target categories
4. **Evaluation**: Evaluate models on the test set (80% of Dataset A) using the optimized mappings

### 1.2 Key Innovation

Unlike traditional approaches that use pre-defined label mappings, we learn the optimal mapping **data-driven** by:
- Training models on their original label systems (preserving semantic richness)
- Using a calibration set to discover which original labels best map to target categories
- Selecting mappings that maximize F1-score on the calibration set

---

## 2. Datasets

### 2.1 Dataset A: Tone Adjustment (Ground Truth)

**Source**: `tone_adjustment.csv` or `datasetA/tone adjustment 1.csv`

**Format**: Wide format with columns:
- `Original`: Original text
- `Polite`: Polite version of the text
- `Professional`: Professional version of the text
- `Casual`: Casual version of the text

**Preprocessing**:
- Transformed from wide to long format
- Each row becomes three rows: one for each tone version
- Labels assigned: 0 (Polite), 1 (Professional), 2 (Casual)
- Original column is ignored

**Splitting**:
- **Calibration Set**: 20% (1,200 samples, stratified by label)
- **Test Set**: 80% (4,800 samples, stratified by label)
- Random seed: 42

**Final Statistics**:
- Total samples: 6,000
- Calibration set: 1,200 (400 per class)
- Test set: 4,800 (1,600 per class)

### 2.2 Dataset B: Tone Analysis

**Source**: `tone_analysis.csv` or `datasetB/tone_v1.txt`

**Format**: Text file with format `text || label.`

**Preprocessing**:
- Parsed from text file using `||` delimiter
- **Original labels preserved** (no pre-defined mapping)
- 30 unique labels: Assertive, Apologetic, Callous, Diplomatic, Witty, Informative, Cautionary, Bitter, Benevolent, Candid, etc.

**Splitting**:
- **Training Set**: 80% (2,684 samples, stratified by label)
- **Validation Set**: 20% (672 samples, stratified by label)
- Random seed: 42

**Final Statistics**:
- Total samples: 3,356
- Training samples: 2,684
- Validation samples: 672
- Number of labels: 30
- Label distribution: Relatively balanced (CV = 0.417)
  - Most frequent: Assertive (214 samples)
  - Least frequent: 15 samples per label

### 2.3 Dataset C: GoEmotions

**Source**: `go_emotions.csv` or `datasetC/goemotions_*.csv`

**Format**: CSV with emotion columns (one-hot encoded)

**Preprocessing**:
- Extracted label from emotion columns using `idxmax()`
- **Original labels preserved** (no pre-defined mapping)
- 28 unique labels: neutral, admiration, approval, annoyance, disapproval, amusement, gratitude, anger, curiosity, disappointment, etc.

**Sampling**:
- **Sampled to 3× Dataset B training size** (target: 8,052 total samples)
- Stratified sampling to preserve label distribution
- Final size: 10,056 samples (8,044 training + 2,012 validation)

**Splitting**:
- **Training Set**: 80% (8,044 samples, stratified by label)
- **Validation Set**: 20% (2,012 samples, stratified by label)
- Random seed: 42

**Final Statistics**:
- Total samples: 10,056
- Training samples: 8,044
- Validation samples: 2,012
- Number of labels: 28
- Label distribution: **Highly imbalanced** (CV = 1.382)
  - Most frequent: neutral (2,109 samples, 26.2% of training set)
  - Least frequent: 19 samples per label
  - Max/min ratio: 111:1

---

## 3. Model Architecture and Training

### 3.1 Model Architecture

**Base Model**: `bert-base-uncased` (Hugging Face Transformers)

**Architecture**: `BertForSequenceClassification`
- Pre-trained BERT encoder (12 layers, 768 hidden dimensions)
- Classification head with `num_labels` output classes
- For Dataset B: `num_labels = 30`
- For Dataset C: `num_labels = 28`

**Tokenizer**: `BertTokenizer.from_pretrained('bert-base-uncased')`
- Max sequence length: 128 tokens
- Truncation: Yes (longer sequences truncated)
- Padding: `max_length` (sequences padded to 128)

### 3.2 Training Configuration

**Framework**: Hugging Face Transformers `Trainer` API

**Hyperparameters**:
- **Epochs**: 3
- **Batch size**: 16 (per device)
- **Learning rate**: Default (5e-5, set by Trainer)
- **Optimizer**: AdamW (default)
- **Warmup steps**: Default (10% of training steps)
- **Weight decay**: 0.01 (default)

**Training Arguments**:
```python
TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',          # Evaluate after each epoch
    save_strategy='epoch',          # Save checkpoint after each epoch
    logging_dir=os.path.join(output_dir, 'logs'),
    logging_steps=100,
    load_best_model_at_end=True,   # Load best model based on eval_accuracy
    metric_for_best_model='eval_accuracy',
    save_total_limit=3,            # Keep only 3 checkpoints
    seed=42
)
```

**Evaluation Metric**: Accuracy (computed on validation set after each epoch)

**Model Selection**: Best model selected based on validation accuracy (`load_best_model_at_end=True`)

### 3.3 Training Process

**Dataset Class**: Custom `ToneDataset` (PyTorch `Dataset`)
- Converts text to tokenized input (input_ids, attention_mask)
- Maps string labels to integer indices using `label_to_idx` dictionary
- Returns dictionary with `input_ids`, `attention_mask`, and `labels`

**Label Mapping**:
- String labels automatically mapped to integer indices (0 to num_labels-1)
- Mapping saved to `label_mapping.json` in model directory
- Format: `{'label_to_idx': {...}, 'idx_to_label': {...}}`

**Training Output**:
- Models saved to: `models/receiver_b_v2/` and `models/receiver_c_v2/`
- Each directory contains:
  - `model.safetensors`: Model weights
  - `config.json`: Model configuration
  - `tokenizer_config.json`: Tokenizer configuration
  - `vocab.txt`: Vocabulary
  - `label_mapping.json`: Original label to index mapping

### 3.4 Training Results

**Model B (Tone Analysis)**:
- Training samples: 2,684
- Validation samples: 672
- Original validation accuracy: 42.26%
- Number of labels: 30

**Model C (GoEmotions)**:
- Training samples: 8,044
- Validation samples: 2,012
- Original validation accuracy: 39.41%
- Number of labels: 28

---

## 4. Label Mapping Optimization

### 4.1 Calibration Set Approach

Instead of using pre-defined label mappings, we learn optimal mappings from data:

1. **Predict on Calibration Set**: Both models predict on Dataset A's calibration set (1,200 samples)
2. **Build Mapping Matrix**: For each original label, count co-occurrence with target labels (0, 1, 2)
3. **Greedy Assignment**: Assign each original label to the target label with highest co-occurrence
4. **Evaluate**: Compute F1-score on calibration set using the mapping
5. **Select Best**: Use mapping that maximizes macro F1-score

### 4.2 Mapping Algorithm

**Step 1: Prediction**
- Run inference on calibration set using trained models
- Get raw predictions (original label indices)

**Step 2: Co-occurrence Matrix**
For each original label `src_label`:
```python
mask = predictions == src_label
true_labels_for_src = true_labels[mask]
counts = {
    0: sum(true_labels_for_src == 0),  # Polite
    1: sum(true_labels_for_src == 1),   # Professional
    2: sum(true_labels_for_src == 2)    # Casual
}
```

**Step 3: Greedy Assignment**
```python
optimal_mapping[src_label] = argmax(counts)  # Target label with max count
```

**Step 4: Evaluation**
- Apply mapping to predictions: `mapped_pred = optimal_mapping[raw_pred]`
- Compute macro F1-score: `f1_score(true_labels, mapped_pred, average='macro')`

### 4.3 Optimal Mappings

**Model B Mapping** (24 original labels → 3 target labels):
- **Polite (0)**: 7 labels (e.g., Aggrieved, Altruistic, Amused, Appreciative, Ardent, Benevolent, Inspirational)
- **Professional (1)**: 6 labels (e.g., Absurd, Apologetic, Assertive, Diplomatic, Informative, Thoughtful)
- **Casual (2)**: 11 labels (e.g., Admiring, Ambivalent, Angry, Animated, Apathetic, Bitter, Callous, Candid, Caustic, Direct, Witty)

**Model C Mapping** (23 original labels → 3 target labels):
- **Polite (0)**: 9 labels (e.g., amusement, caring, confusion, disapproval, joy, optimism, remorse, sadness, surprise)
- **Professional (1)**: 4 labels (e.g., approval, excitement, gratitude, neutral)
- **Casual (2)**: 10 labels (e.g., admiration, anger, annoyance, curiosity, desire, disappointment, disgust, fear, love, realization)

**Calibration Set Performance**:
- Model B: Accuracy 52.83%, Macro F1 0.5189
- Model C: Accuracy 47.75%, Macro F1 0.4777

---

## 5. Evaluation

### 5.1 Evaluation Protocol

1. **Load Optimal Mapping**: Load mapping from calibration set optimization
2. **Predict on Test Set**: Run inference on Dataset A test set (4,800 samples)
3. **Apply Mapping**: Map raw predictions to target labels using optimal mapping
4. **Compute Metrics**: Calculate accuracy, precision, recall, F1-score, confusion matrix

### 5.2 Metrics

**Metrics Computed**:
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted mean of F1-scores for each class
- **Weighted F1**: F1-score weighted by class support
- **Per-class Metrics**: Precision, Recall, F1-score for each class (Polite, Professional, Casual)
- **Confusion Matrix**: 3×3 matrix showing predicted vs. true labels

### 5.3 Final Results

#### Model B (Tone Analysis)

**Test Set Performance**:
- **Accuracy**: 54.58% (2,620 / 4,800 correct)
- **Macro F1**: 0.5337
- **Weighted F1**: 0.5337

**Per-Class Performance**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Polite | 0.50 | 0.58 | 0.54 | 1,600 |
| Professional | 0.57 | 0.58 | 0.57 | 1,600 |
| Casual | 0.57 | 0.48 | 0.52 | 1,600 |

**Confusion Matrix**:

<image>
confusion_matrix_model_b_v2.png
</image>

#### Model C (GoEmotions)

**Test Set Performance**:
- **Accuracy**: 46.50% (2,232 / 4,800 correct)
- **Macro F1**: 0.4635
- **Weighted F1**: 0.4635

**Per-Class Performance**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Polite | 0.41 | 0.49 | 0.45 | 1,600 |
| Professional | 0.50 | 0.53 | 0.51 | 1,600 |
| Casual | 0.51 | 0.37 | 0.43 | 1,600 |

**Confusion Matrix**:

<image>
confusion_matrix_model_c_v2.png
</image>

---

## 6. Analysis: Why Model B Outperforms Model C

Despite having **3× fewer training samples** (2,684 vs 8,044), Model B achieves **8.08% higher accuracy** and **0.0702 higher macro F1** than Model C. This counterintuitive result can be explained by several factors:

### 6.1 Label Distribution Imbalance (Primary Factor)

**Dataset B**:
- Coefficient of Variation (CV): 0.417
- Max/min ratio: 14.27 (214/15)
- Relatively balanced distribution

**Dataset C**:
- Coefficient of Variation (CV): 1.382 (**3.3× higher than B**)
- Max/min ratio: 111.00 (2,109/19)
- **Neutral label dominates**: 26.2% of all training samples
- Top 3 labels account for 43.3% of samples

**Impact**:
- Model C tends to predict high-frequency labels (especially "neutral")
- Low-frequency labels (e.g., "disgust" with only 19 samples) are under-trained
- Class imbalance leads to biased predictions and lower overall performance

### 6.2 Domain Alignment (Secondary Factor)

**Dataset B (Tone Analysis)**:
- Labels: Assertive, Apologetic, Diplomatic, Witty, Informative, Cautionary, etc.
- **Directly targets tone/style characteristics**
- Semantic distance to target categories (Polite, Professional, Casual) is small

**Dataset C (GoEmotions)**:
- Labels: neutral, admiration, approval, annoyance, anger, sadness, etc.
- **Primarily emotion classification labels**
- Semantic distance to target categories (tone adjustment) is large

**Impact**:
- Dataset B's labels map more naturally to target categories
- Dataset C requires bridging a semantic gap from "emotion" to "tone"
- Even with more samples, domain mismatch limits performance gains

### 6.3 Mapping Quality

**Model B Mapping**:
- 24 original labels → 3 target labels
- Distribution: Polite(7), Professional(6), Casual(11) - relatively balanced

**Model C Mapping**:
- 23 original labels → 3 target labels
- Distribution: Polite(9), Professional(4), Casual(10)
- Professional category has fewer source labels (only 4), potentially limiting mapping quality

### 6.4 Key Insight

**Sample Quality > Sample Quantity**

In transfer learning scenarios:
- **Relevant domain with fewer samples** (Dataset B) > **Irrelevant domain with more samples** (Dataset C)
- Domain alignment and label balance are more critical than raw sample count
- The 3× sample advantage of Dataset C is offset by domain mismatch and severe class imbalance

---

## 7. Implementation Details

### 7.1 Software and Libraries

- **Python**: 3.x
- **PyTorch**: For deep learning framework
- **Transformers**: Hugging Face library (version with `BertTokenizer`, `BertForSequenceClassification`, `Trainer`)
- **scikit-learn**: For metrics (`accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`)
- **pandas**: For data manipulation
- **numpy**: For numerical operations

### 7.2 Reproducibility

- **Random seed**: 42 (used for data splitting and sampling)
- **Stratified splitting**: Ensures balanced label distribution across splits
- **Deterministic operations**: All random operations use fixed seed

---

## 8. Conclusions

### 8.1 Main Findings

1. **Data-driven label mapping** successfully adapts models trained on diverse label systems to target categories, achieving reasonable performance without manual mapping design.

2. **Domain alignment matters more than sample count**: Model B (2,684 samples) outperforms Model C (8,044 samples) due to better domain match and balanced label distribution.

3. **Class imbalance is a critical factor**: Dataset C's severe imbalance (CV=1.382, max/min=111:1) significantly degrades performance despite having more samples.

4. **Calibration set approach is effective**: Using 20% of target data for mapping optimization provides a good balance between optimization and evaluation.

### 8.2 Limitations

1. **Simple greedy mapping**: Current approach uses greedy assignment; more sophisticated methods (e.g., integer linear programming) might yield better mappings.

2. **Fixed calibration/test split**: 20/80 split is arbitrary; different ratios might affect results.

3. **No ensemble**: Models B and C are evaluated separately; ensemble methods could potentially improve performance.

4. **Limited hyperparameter tuning**: Training uses default hyperparameters; tuning might improve results.

### 8.3 Future Work

1. **Advanced mapping methods**: Explore optimal assignment algorithms (e.g., Hungarian algorithm, ILP) instead of greedy approach.

2. **Data balancing**: Apply techniques (SMOTE, undersampling) to Dataset C to address class imbalance.

3. **Domain adaptation**: Use domain adaptation techniques (e.g., adversarial training) to bridge domain gap for Dataset C.

4. **Ensemble methods**: Combine predictions from Models B and C using voting or weighted averaging.

5. **Hyperparameter optimization**: Conduct systematic hyperparameter search for both models.

---

## 9. References

- **BERT**: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.

- **Hugging Face Transformers**: Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020.

- **GoEmotions Dataset**: Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.

---

**Report Generated**: November 2024  
**Project**: Cross-Domain Tone Classification  
**Method**: Data-Driven Label Mapping with BERT

