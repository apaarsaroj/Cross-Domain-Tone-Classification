# Performance Difference Analysis: Dataset C vs Dataset B

## Performance Comparison Summary

| Metric | Model B | Model C | Difference |
|--------|---------|---------|------------|
| Training Samples | 2,684 | 8,044 | C is 3.0x B |
| Number of Labels | 30 | 28 | Similar |
| Validation Accuracy (Original) | 42.26% | 39.41% | B higher by 2.85% |
| Test Set Accuracy | 48.33% | 46.50% | B higher by 1.83% |
| Test Set Macro F1 | 0.4833 | 0.4635 | B higher by 0.0198 |

## Key Findings

### 1. Highly Imbalanced Label Distribution (Primary Cause)

**Dataset B:**
- Label distribution coefficient of variation (CV): 0.417
- Max/min sample ratio: 14.27 (214/15)
- Relatively balanced distribution, all labels have sufficient training samples

**Dataset C:**
- Label distribution coefficient of variation (CV): 1.382 (3.3x that of B)
- Max/min sample ratio: 111.00 (2109/19)
- **Neutral label accounts for 26.2%**, top 3 labels account for 43.3%
- Severe class imbalance problem

**Impact:**
- Model tends to predict high-frequency labels (e.g., neutral)
- Low-frequency labels (e.g., disgust with only 19 samples) are under-trained
- Leads to overall performance degradation

### 2. Domain Matching Differences

**Dataset B (Tone Analysis):**
- Label semantics: Assertive, Apologetic, Diplomatic, Witty, Informative, Cautionary, etc.
- **Directly targets tone/style features**
- Closer semantic distance to target task (Polite, Professional, Casual)

**Dataset C (GoEmotions):**
- Label semantics: neutral, admiration, approval, annoyance, anger, sadness, etc.
- **Primarily emotion classification labels**
- Further semantic distance from target task (tone adjustment)

**Impact:**
- Dataset B labels are easier to map to target categories
- Dataset C needs to bridge the semantic gap from "emotion" to "tone"
- Even with more samples, domain mismatch limits performance improvement

### 3. Label Mapping Quality

**Dataset B Mapping:**
- 24 original labels → 3 target labels
- Mapping distribution: Polite(7), Professional(6), Casual(11)
- Relatively balanced mapping

**Dataset C Mapping:**
- 23 original labels → 3 target labels
- Mapping distribution: Polite(9), Professional(4), Casual(10)
- Professional category has fewer mappings (only 4 original labels)

**Impact:**
- Dataset C's Professional category may have lower mapping quality
- From confusion matrix, Professional category performs best (F1=0.51), but this is because it is mapped to high-frequency labels like neutral

### 4. Average Samples per Label

Although Dataset C has an average of 287.3 samples per label (vs B's 89.5):
- **Sample quality matters more**: Samples with low domain match provide limited help even if numerous
- **Imbalance offsets advantage**: Although average samples are more, the distribution is highly uneven, and low-frequency labels are still under-trained

### 5. Model Complexity

Both models use the same training configuration:
- Epochs: 3
- Batch size: 16
- Same BERT architecture

However:
- Dataset C has 28 labels (vs B's 30), similar classification task complexity
- Due to label imbalance, the model needs more training to learn low-frequency labels, but the number of training epochs is the same

## Conclusion

**Why does Dataset C have more samples but worse performance?**

1. **Primary cause: Highly imbalanced label distribution**
   - Neutral label accounts for 26.2%, causing the model to favor predicting high-frequency labels
   - Low-frequency labels are under-trained (minimum label has only 19 samples)

2. **Secondary cause: Low domain matching**
   - Dataset C is an emotion classification dataset, with greater semantic distance from the tone adjustment task
   - Dataset B directly targets tone features, better matching the target task

3. **Sample quality > Sample quantity**
   - Even though Dataset C has 3x the samples, domain mismatch and label imbalance offset the quantity advantage
   - In transfer learning scenarios, a small number of high-quality samples from a related domain > a large number of samples from an unrelated domain
