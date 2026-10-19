# Chest X-Ray Multi-Label Classification

A deep learning project for multi-label chest X-ray classification using transfer learning with ResNet-18 and PyTorch. The project includes a complete training and evaluation pipeline, along with an interactive Streamlit application for image-level predictions.

## Overview

The model predicts the presence of 14 findings from a chest X-ray image. Unlike single-class classification, this is a multi-label task: multiple findings can be predicted for the same image.

**Key features**
- Transfer learning with a pretrained ResNet-18
- Patient-level train, validation, and test splits to reduce data leakage
- Multi-label classification using 14 output probabilities
- Model checkpoint selection based on validation macro ROC-AUC
- Per-finding ROC and precision-recall curves
- Interactive Streamlit app for uploading X-rays and viewing predictions

## Model Performance

Evaluation on the held-out test set:

| Metric | Score |
|---|---:|
| Macro ROC-AUC | 0.8151 |
| Macro Average Precision | 0.2367 |
| Macro F1-score | 0.3036 |

The best checkpoint was selected at epoch 4, with a validation macro ROC-AUC of 0.8215.

These metrics summarize performance across the 14 findings. Results vary by finding, and the F1-score depends on the selected decision thresholds.

## Predicted Findings

The model predicts the following labels:

Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia.

## Tech Stack

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Pillow

## Project Structure

```text
chest-xray-multilabel/
├── app.py
├── src/
│   ├── dataset.py
│   └── model.py
├── models/
│   ├── best_resnet18_extended.pth
│   └── thresholds.json
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── figures/
│   ├── predictions/
│   └── test_per_disease_summary.csv
├── test_evaluation.py
├── plot_test_curves.py
├── plot_training_history.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/ibarbary/chest-xray-multilabel.git
cd chest-xray-multilabel
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the app

```bash
python -m streamlit run app.py
```

Upload a chest X-ray image to view the model's predicted scores for the 14 findings.

**Note:** The trained model checkpoint and threshold configuration must be available at the paths expected by the app. The original dataset is not included in this repository.

## Dataset and Methodology

The project uses a chest X-ray dataset with 14 finding labels. Images are split into training, validation, and test sets at the patient level, so images belonging to the same patient do not cross between splits.

The model uses a pretrained ResNet-18 backbone with a 14-output classification head. Model selection is based on validation macro ROC-AUC, and the final evaluation is performed on the held-out test set.

## Limitations and Disclaimer

This is an educational and research project, not a clinically validated medical device. Predictions are model estimates and must not be used to diagnose conditions, guide treatment, or replace professional medical evaluation. Performance may vary with image sources, acquisition protocols, and patient populations.

## Future Improvements

- Improve per-finding classification performance and threshold selection
- Evaluate calibration and robustness across datasets
- Deploy the Streamlit application for public demonstration
- Add automated testing and experiment tracking
