# Experiment summary

## Dataset
Makerere AI Lab Beans dataset, version 0.1.0; 1,295 images total (1,034 train, 133 validation, 128 test). Classes: Angular Leaf Spot, Bean Rust, Healthy.

## Experiment A — Baseline CNN
- Epochs completed: 30
- Best validation-loss epoch: 28
- Best validation loss: 0.34343
- Validation accuracy at best-loss epoch: 87.22%
- Test accuracy: 83.59%
- Test loss: 0.3849
- Macro F1: 0.83465
- Weighted F1: 0.83420

Test confusion matrix (actual rows, predicted columns; class order ALS, Bean Rust, Healthy):
~~~text
[[38,  4,  1],
 [ 8, 31,  4],
 [ 1,  3, 38]]
~~~

## Experiment B — Augmented CNN
Augmentation: RandomFlip(horizontal), RandomRotation(0.10), RandomZoom(0.10), RandomContrast(0.10).
- Epochs completed: 30
- Best validation-loss epoch: 24
- Best validation loss: 0.50764
- Validation accuracy at best-loss epoch: 81.20%
- Test accuracy: 78.12%
- Test loss: 0.4901
- Macro F1: 0.77612
- Weighted F1: 0.77566

Test confusion matrix:
~~~text
[[36,  3,  4],
 [10, 26,  7],
 [ 0,  4, 38]]
~~~

## Interpretation
The baseline CNN outperformed the augmented CNN on this held-out test split:
- Accuracy difference: 5.47 percentage points
- Macro F1 difference: approximately 5.85 percentage points
- Bean Rust remained the most challenging class for both models.

The baseline was selected as the final model for this project. This conclusion is limited to this dataset split and the two recorded implementations. The baseline and augmented definitions used different Conv2D padding settings (valid by default versus explicit same), which should be disclosed as an experimental limitation.
