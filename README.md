# Deep Learning Based Bean Leaf Disease Classification Using CNN

**NIELIT AIDA Capstone Project**

A three-class image-classification project using a Convolutional Neural Network (CNN) to classify bean leaf images as Angular Leaf Spot, Bean Rust, or Healthy.

## Project status

- Baseline CNN training: completed (30 epochs; best checkpoint selected using validation loss).
- Augmented CNN training: completed (30 epochs; best checkpoint selected using validation loss).
- Independent test evaluation: completed for both models.
- Streamlit application: implemented in app.py.
- Final model artifact: stored separately in the project owner's Google Drive during development. Place it at models/final_bean_leaf_disease_cnn.keras before running the app. The binary model is not included in this source-code commit.

## Dataset

Makerere AI Lab Beans Dataset (TensorFlow Datasets name: beans, version 0.1.0).

- Smartphone field images of bean leaves.
- Classes: Angular Leaf Spot, Bean Rust, and Healthy.
- Dataset description identifies NaCRRI experts in Uganda as annotators and Makerere AI Research Lab as the collecting lab.
- Original images: 500 x 500 RGB, uint8.
- Model input: 224 x 224 RGB, float32, pixel values scaled to [0, 1].

| Split | Angular Leaf Spot | Bean Rust | Healthy | Total |
|---|---:|---:|---:|---:|
| Train | 345 | 348 | 341 | 1,034 |
| Validation | 44 | 45 | 44 | 133 |
| Test | 43 | 43 | 42 | 128 |
| **Total** | **432** | **436** | **427** | **1,295** |

Dataset references:
- https://www.tensorflow.org/datasets/catalog/beans
- https://github.com/AI-Lab-Makerere/ibean

Dataset image files are not included in this repository.

## Model experiments

### Baseline CNN

Four convolution/max-pooling blocks (32, 64, 128, and 256 filters), Global Average Pooling, a 128-unit dense layer, Dropout (0.4), and a three-unit softmax output.

### Augmented CNN

The second experiment added training-time augmentation:
- Random horizontal flip
- Random rotation (0.10)
- Random zoom (0.10)
- Random contrast (0.10)

Both models have 421,699 trainable parameters. The recorded model definitions differ in Conv2D padding: the baseline used the Keras default (valid), while the augmented model used same padding. This implementation difference should be considered when interpreting the comparison.

## Test-set results

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
~~~text
[[38,  4,  1],
 [ 8, 31,  4],
 [ 1,  3, 38]]
~~~

Augmented CNN:
~~~text
[[36,  3,  4],
 [10, 26,  7],
 [ 0,  4, 38]]
~~~

The baseline model is selected as the final model because it achieved higher test accuracy and F1-scores and lower test loss in this experiment. This result is limited to this held-out split; it does not establish that baseline CNNs outperform augmentation on all bean-leaf datasets.

## Run the Streamlit app

1. Install dependencies:
   ~~~bash
   pip install -r requirements.txt
   ~~~
2. Place the trained model at:
   ~~~text
   models/final_bean_leaf_disease_cnn.keras
   ~~~
   The model is currently stored separately in the project owner's Google Drive and is not included in this commit.
3. Launch:
   ~~~bash
   streamlit run app.py
   ~~~

The app preprocesses images as RGB, resizes them to 224 x 224 using bilinear interpolation, converts to float32, and scales pixel values to [0, 1].

## Repository layout

~~~text
.
|-- app.py
|-- requirements.txt
|-- models/
|   |-- README.md
|-- results/
|   |-- test_metrics_comparison.csv
|   |-- classwise_f1_comparison.csv
|   |-- confusion_matrices.json
|-- reports/
|   |-- experiment_summary.md
~~~

## Important limitation

The displayed confidence is the model's softmax score, not a calibrated probability or expert-confirmed diagnosis. The application provides preliminary AI classification only. Agricultural treatment decisions should be verified with an agricultural expert.
