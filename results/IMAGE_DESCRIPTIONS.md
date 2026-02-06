# Image Descriptions

This document describes all visualization images generated for the project.

## Training Curves

### `training_curves_receiver_b_v2.png`
**Description**: Training progress visualization for Model B (Tone Analysis dataset).

**Content**:
- **Top Left**: Training loss over training steps. Shows how the model's loss decreases during training.
- **Top Right**: Validation loss over training steps. Lower values indicate better generalization.
- **Bottom Left**: Validation accuracy over training steps. Higher values (closer to 1.0) indicate better performance.
- **Bottom Right**: Learning rate schedule. Shows how the learning rate changes during training.

**Interpretation**: 
- Training loss should decrease steadily, indicating the model is learning.
- Validation loss should also decrease; if it starts increasing while training loss decreases, this indicates overfitting.
- Validation accuracy should increase over epochs, ideally reaching a plateau near the end.
- Learning rate typically decreases over time to allow fine-tuning.

### `training_curves_receiver_c_v2.png`
**Description**: Training progress visualization for Model C (GoEmotions dataset).

**Content**: Same structure as Model B's training curves.

**Interpretation**: Compare with Model B to understand differences in training dynamics.

---

## Label Distributions

### `label_distribution_model_b.png`
**Description**: Distribution of original labels in Model B's training dataset.

**Content**: Bar chart showing the count of each original label in the training set.

**Interpretation**:
- Shows class imbalance in the original dataset.
- Labels with very high counts may dominate the model's predictions.
- Labels with very low counts may be under-trained.

### `label_distribution_model_c.png`
**Description**: Distribution of original labels in Model C's training dataset.

**Content**: Bar chart showing the count of each original label in the training set.

**Interpretation**: 
- Compare with Model B to see differences in label distribution.
- Model C's severe imbalance (visible in the chart) explains its lower performance despite having more samples.

---

## Optimal Label Mappings

### `optimal_mapping_model_b.png`
**Description**: Visualization of how Model B's original labels are mapped to the 3 target categories (Polite, Professional, Casual).

**Content**: Bar chart showing the number of original labels mapped to each target category.

**Interpretation**:
- Shows the distribution of source labels across target categories.
- A balanced distribution (similar bar heights) is generally better.
- This mapping was optimized on the calibration set to maximize F1-score.

### `optimal_mapping_model_c.png`
**Description**: Visualization of how Model C's original labels are mapped to the 3 target categories.

**Content**: Bar chart showing the number of original labels mapped to each target category.

**Interpretation**: Compare with Model B to see differences in mapping strategies.

---

## Confusion Matrices

### `confusion_matrix_model_b_v2.png`
**Description**: Confusion matrix for Model B's predictions on Dataset A test set.

**Content**: Heatmap showing predicted labels (columns) vs. true labels (rows). Diagonal elements represent correct predictions.

**Interpretation**:
- **Diagonal values**: Correct predictions (higher is better).
- **Off-diagonal values**: Misclassifications. Patterns show which classes are confused with each other.
- **Row sums**: Total samples per true label (should be balanced: 1,600 each).
- **Column sums**: Total predictions per predicted label.

**Key Insights**:
- Model B achieves 54.58% accuracy.
- The matrix shows which tone categories are most confused (e.g., Polite vs. Professional).

### `confusion_matrix_model_c_v2.png`
**Description**: Confusion matrix for Model C's predictions on Dataset A test set.

**Content**: Same structure as Model B's confusion matrix.

**Interpretation**:
- Model C achieves 46.50% accuracy.
- Compare with Model B to identify differences in error patterns.
- Off-diagonal patterns reveal systematic misclassifications.

---

## Summary

All images are saved in the `results/` directory and can be used in the final report. The training curves show model learning dynamics, label distributions reveal data characteristics, optimal mappings show the label alignment strategy, and confusion matrices quantify model performance and error patterns.


