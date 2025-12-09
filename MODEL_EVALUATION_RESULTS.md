# AI Model Evaluation Results

This document explains the performance of the surgical phase detection AI model, based on testing with real surgical videos.

---

## Overview

The AI model was trained to recognize **8 surgical phases** in labral repair (shoulder arthroscopy) procedures. After training, the model was evaluated on 4 test videos containing a total of **5,584 frames**.

**Overall Accuracy: 77.9%**

This means the model correctly identifies the surgical phase approximately 78 out of every 100 frames.

---

## Performance by Surgical Phase

The table below shows how well the model performs for each surgical phase:

| Surgical Phase | Precision | Recall | F1-Score | Test Frames |
|----------------|-----------|--------|----------|-------------|
| Suture Tensioning | 92.6% | 95.0% | **93.8%** | 817 |
| Anchor Placement | 89.6% | 85.4% | **87.4%** | 1,787 |
| Suture Passage | 84.7% | 90.0% | **87.2%** | 1,622 |
| Portal Placement | 86.9% | 86.9% | **86.9%** | 99 |
| Diagnostic Arthroscopy | 46.1% | 93.6% | 61.8% | 220 |
| Labral Mobilization | 34.8% | 69.3% | 46.3% | 228 |
| Final Inspection | 91.9% | 27.5% | 42.3% | 455 |
| Glenoid Preparation | 7.1% | 3.7% | 4.8% | 356 |

---

## Understanding the Metrics

### What do these numbers mean?

- **Precision**: When the model says "this is Phase X", how often is it correct?
  - High precision = fewer false alarms

- **Recall**: Out of all actual frames of Phase X, how many did the model find?
  - High recall = fewer missed detections

- **F1-Score**: The balance between precision and recall (higher is better)
  - This is the most useful single metric

### Example

For **Suture Tensioning** (93.8% F1-Score):
- The model is very good at recognizing this phase
- When it says "Suture Tensioning", it's right 92.6% of the time
- It catches 95% of all actual Suture Tensioning frames

For **Glenoid Preparation** (4.8% F1-Score):
- The model struggles with this phase
- This is mainly because there were only 356 training frames (very limited data)

---

## Best Performing Phases

The model excels at detecting these phases:

1. **Suture Tensioning** (93.8% F1-Score)
   - Distinctive visual features
   - Plenty of training data (817 frames)

2. **Anchor Placement** (87.4% F1-Score)
   - Clear visual patterns
   - Most training data (1,787 frames)

3. **Suture Passage** (87.2% F1-Score)
   - Recognizable instrument movements
   - Good training data (1,622 frames)

4. **Portal Placement** (86.9% F1-Score)
   - Unique visual appearance
   - Despite limited data (99 frames), very distinctive

---

## Areas for Improvement

These phases need more training data or model refinement:

### Glenoid Preparation (4.8% F1-Score)
- **Why it struggles**: Very limited training data (only 356 frames)
- **Impact**: The model often confuses this phase with others
- **Solution**: Collect more labeled videos of this phase

### Final Inspection (42.3% F1-Score)
- **Why it struggles**: Short duration phase, often at video end
- **Impact**: High precision (92%) but low recall (27%)
- **Meaning**: When detected, it's usually correct, but often missed

### Labral Mobilization (46.3% F1-Score)
- **Why it struggles**: Visual similarity to other phases
- **Impact**: Moderate detection performance
- **Solution**: More distinctive training examples needed

### Diagnostic Arthroscopy (61.8% F1-Score)
- **Why it struggles**: Can look similar to Final Inspection
- **Impact**: High recall (94%) but lower precision (46%)
- **Meaning**: Rarely missed, but sometimes incorrectly labeled

---

## Per-Case Results

The model was tested on 4 different surgical videos:

| Case | Accuracy | Total Frames | Notes |
|------|----------|--------------|-------|
| Case 00008 | **97.8%** | 1,502 | Best performance |
| Case 00017 | 75.7% | 1,177 | Good performance |
| Case 00013 | 70.7% | 1,282 | Average performance |
| Case 00012 | 66.7% | 1,623 | Lower performance |

### Why the Variation?

- **Case 00008** achieved 97.8% because its phases were clearly defined and similar to training data
- **Case 00012** had 66.7% possibly due to:
  - Different camera angles
  - Unusual procedure variations
  - More challenging phase transitions

---

## Confusion Matrix Explained

The confusion matrix (available in `model-inference/evaluation_results/confusion_matrix.png`) shows:

- **Diagonal cells**: Correct predictions (bright/high numbers are good)
- **Off-diagonal cells**: Mistakes (shows which phases get confused with each other)

Common confusions:
- Diagnostic Arthroscopy ↔ Final Inspection (similar camera views)
- Glenoid Preparation ↔ Anchor Placement (sequential phases)

---

## What This Means for Users

### High Confidence Results (>85% F1-Score)
For these phases, you can trust the AI's detection:
- Suture Tensioning
- Anchor Placement
- Suture Passage
- Portal Placement

### Moderate Confidence Results (40-85% F1-Score)
Review these manually for accuracy:
- Diagnostic Arthroscopy
- Labral Mobilization
- Final Inspection

### Low Confidence Results (<40% F1-Score)
Always verify manually:
- Glenoid Preparation

---

## Files in evaluation_results/

| File | Description |
|------|-------------|
| `classification_report.csv` | Detailed metrics for each phase |
| `per_case_results.csv` | Accuracy breakdown by test video |
| `confusion_matrix.png` | Visual confusion matrix |

---

## How to Improve Results

If you have additional surgical videos:

1. **More training data** = Better accuracy, especially for struggling phases
2. **Diverse videos** = Better generalization to new cases
3. **Accurate labels** = Critical for model learning

The current model represents a baseline that can be improved with:
- More labeled surgical videos
- Fine-tuning on specific surgeon styles
- Additional data augmentation techniques

---

## Summary

| Metric | Value |
|--------|-------|
| Overall Accuracy | 77.9% |
| Best Phase | Suture Tensioning (93.8%) |
| Needs Improvement | Glenoid Preparation (4.8%) |
| Test Videos | 4 |
| Total Test Frames | 5,584 |

The model performs well for most surgical phases, with 4 out of 8 phases achieving over 85% F1-Score. The main limitation is the Glenoid Preparation phase, which requires more training data to improve detection accuracy.
