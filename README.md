# Deep Learning Based Bean Leaf Disease Classification Using CNN

**NIELIT AIDA Capstone Project**  
**Student:** Madhav Raman · **Enrollment No.:** 2001191  
**Programme:** AIDA – Artificial Intelligence Development Associate  
**Report submission date:** 12 November 2026

A three-class image-classification project using a Convolutional Neural Network (CNN) to classify bean-leaf images as Angular Leaf Spot, Bean Rust, or Healthy.

## Project contents

- `train_model.py` — consolidated training and evaluation script for the baseline and augmented CNN experiments.
- `training_requirements.txt` — dependencies for model training and evaluation.
- `app.py` — Streamlit application for image upload and model inference.
- `requirements.txt` — dependencies for the Streamlit application.
- `models/final_bean_leaf_disease_cnn.keras` — selected baseline CNN model used by the Streamlit application.
- `results/` — recorded test metrics, class-wise F1 comparison, and confusion matrices.
- `reports/experiment_summary.md` — experiment summary.

## Important note about the training script

The supplied `train_model.py` is a consolidated standalone reproduction script aligned with the documented architecture and experiment settings. It is not an export of the original interactive Google Colab notebook. The historic results reported below are based on the saved project experiment outputs. A fresh run is a new reproduction run and may produce different metrics.

## Dataset

Makerere AI Lab Beans Dataset (TensorFlow Datasets name: `beans`, version 0.1.0).

- Smartphone field images of bean leaves.
- Classes: Angular Leaf Spot, Bean Rust, and Healthy.
- Dataset documentation identifies NaCRRI experts in Uganda as annotators and Makerere AI Research Lab as the collecting lab.
- Original images: 500 × 500 RGB, uint8.
- Model input: 224 × 224 RGB, float32, pixel values scaled to [0, 1].

| Split | Angular Leaf Spot | Bean Rust | Healthy | Total |
|---|---:|---:|---:|---:|
| Train | 345 | 348 | 341 | 1,034 |
| Validation | 44 | 45 | 44 | 133 |
| Test | 43 | 43 | 42 | 128 |
| **Total** | **432** | **436** | **427** | **1,295** |

Dataset references:
- https://www.tensorflow.org/datasets/catalog/beans
- https://github.com/AI-Lab-Makerere/ibean

Dataset image files are not included in this repository. Obtain the dataset from the cited source and arrange it as follows before training:

```text
dataset/
  train/
    angular_leaf_spot/
    bean_rust/
    healthy/
  validation/
    angular_leaf_spot/
    bean_rust/
    healthy/
  test/
    angular_leaf_spot/
    bean_rust/
    healthy/
```

## Model experiments

### Baseline CNN

Four convolution/max-pooling blocks (32, 64, 128, and 256 filters), Global Average Pooling, a 128-unit dense layer, Dropout (0.4), and a three-unit softmax output.

### Augmented CNN

The second recorded experiment added training-time augmentation:
- Random horizontal flip
- Random rotation (0.10)
- Random zoom (0.10)
- Random contrast (0.10)

Both models have 421,699 parameters. The recorded model definitions differ in Conv2D padding: the baseline used the Keras default (`valid`), while the augmented model used `same` padding. This implementation difference should be considered when interpreting the comparison.

## Recorded test-set results

| Metric | Baseline CNN | Augmented CNN |
|---|---:|---:|
| Test accuracy | **83.59%** | 78.12% |
| Test loss | **0.3849** | 0.4901 |
| Macro F1-score | **83.47%** | 77.61% |
| Weighted F1-score | **83.42%** | 77.57% |

The baseline correctly classified 107 of 128 test images; the augmented model correctly classified 100 of 128.

### Class-wise F1-score

| Class | Baseline | Augmented |
|---|---:|---:|
| Angular Leaf Spot | 84.44% | 80.90% |
| Bean Rust | 76.54% | 68.42% |
| Healthy | 89.41% | 83.52% |

Confusion matrices (rows = actual class; columns = predicted class; order: Angular Leaf Spot, Bean Rust, Healthy):

Baseline CNN:
```text
[[38,  4,  1],
 [ 8, 31,  4],
 [ 1,  3, 38]]
```

Augmented CNN:
```text
[[36,  3,  4],
 [10, 26,  7],
 [ 0,  4, 38]]
```

The baseline model was selected as the final model because it achieved higher test accuracy and F1-scores and lower test loss in this experiment. This finding is limited to the held-out split and the two recorded configurations; it does not establish that baseline CNNs outperform augmentation generally.

## Train the models

Use Python with the packages in `training_requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r training_requirements.txt
python train_model.py --dataset-dir dataset --output-dir training_outputs --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 42
```

To run only one experiment, add for example:

```bash
python train_model.py --experiments baseline
```

The selected baseline model is written to:

```text
training_outputs/models/final_bean_leaf_disease_cnn.keras
```

The script also writes model checkpoints, epoch histories, classification reports, confusion matrices, and a JSON training summary under `training_outputs/`.

## Run the Streamlit application

Install the application dependencies:

```bash
pip install -r requirements.txt
```

The selected baseline model is included in the repository at:

```text
models/final_bean_leaf_disease_cnn.keras
```

Launch the application from the repository root:

```bash
streamlit run app.py
```

The app preprocesses images as RGB, resizes them to 224 × 224 using bilinear interpolation, converts to float32, and scales pixel values to [0, 1].

## Important limitation

The displayed confidence is the model's softmax score, not a calibrated probability or an expert-confirmed diagnosis. The application provides preliminary AI classification only. Agricultural treatment decisions should be verified by an agricultural expert.
