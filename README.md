# Chest X-Ray Multi-Label Classification

A deep learning project for multi-label chest X-ray classification using **PyTorch, transfer learning, and ResNet-18**. The project includes a patient-level data splitting pipeline, model training and evaluation, per-disease performance analysis, and an interactive Streamlit application for inference.

**Test Macro ROC-AUC: 0.8151** | **14 predicted findings** | **ResNet-18**

> **Disclaimer:** This project is for educational and research purposes only. It is not a clinically validated diagnostic tool.

## Overview

Chest X-ray interpretation can involve multiple findings in a single image. This project approaches the task as **multi-label classification**, predicting a probability for each of 14 findings rather than assigning one mutually exclusive class.

### Key Features

- Transfer learning with a pretrained ResNet-18 backbone
- Patient-level train, validation, and test splits to reduce data leakage
- Multi-label prediction with 14 output probabilities
- Model selection using validation macro ROC-AUC
- Test-set evaluation using ROC-AUC, average precision, and F1-score
- Per-disease ROC and precision-recall curves
- Training history and performance visualizations
- Interactive Streamlit app for uploading a chest X-ray and inspecting predictions

## Model Performance

Results on the held-out test set:

| Metric | Score |
|---|---:|
| Macro ROC-AUC | **0.8151** |
| Macro Average Precision | **0.2367** |
| Macro F1-score | **0.3036** |

The selected checkpoint was from **epoch 4**, with a validation macro ROC-AUC of **0.8215**.

Macro metrics average performance across the 14 findings, giving each finding equal weight. Average precision and F1-score are affected by class imbalance and decision thresholds; these results should be interpreted in the context of the dataset and evaluation setup.

### Evaluation Visualizations

The repository includes ROC curves, precision-recall curves, training history, and per-disease performance summaries in `reports/`.

## Predicted Findings

The model predicts probabilities for these 14 findings:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Data Processing:** NumPy, Pandas
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib
- **Application:** Streamlit
- **Image Processing:** Pillow

## Project Structure

```text
Chest-Xray-Classifier/
├── app.py
├── train.py
├── prepare_splits.py
├── src/
│   ├── dataset.py
│   └── model.py
├── models/
│   └── thresholds.json
├── reports/
│   ├── figures/
│   ├── training_history.csv
│   └── test_per_disease_summary.csv
├── plot_test_curves.py
├── plot_training_history.py
├── requirements.txt
├── .gitignore
└── README.md
```

The trained checkpoint is hosted separately on Hugging Face because it exceeds GitHub's standard file-size limit. The dataset and generated test predictions are not included in this repository.

## Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/ibarbary/Chest-Xray-Classifier.git
cd Chest-Xray-Classifier
```

### 2. Download the trained checkpoint

Download `best_resnet18_extended.pth` from the [Hugging Face model repository](https://huggingface.co/Baarbaary/chest-xray-classifier-resnet18) and place it in the `models/` directory.

The expected local path is:

```text
models/best_resnet18_extended.pth
```

Keep `models/thresholds.json` in place; the application uses the threshold configuration included in this repository.

### 3. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python directly using `.venv\Scripts\python.exe`.

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Upload a chest X-ray image to inspect the model's predicted probabilities for the 14 findings.

**Note:** The inference application requires the trained checkpoint and threshold configuration. Reproducing training and evaluation additionally requires the original dataset and the appropriate prepared data splits.

## Dataset and Methodology

The project uses a chest X-ray dataset with 14 finding labels.

### Patient-Level Data Splitting

The `prepare_splits.py` script creates training, validation, and test splits by **patient ID**, rather than randomly splitting individual images. This helps prevent images from the same patient appearing in multiple splits and reduces the risk of patient-level data leakage.

### Model and Training

A pretrained ResNet-18 is adapted with a 14-output classification head. The model produces one score per finding, supporting multiple simultaneous positive predictions.

The best checkpoint is selected based on validation macro ROC-AUC. Final performance is then measured on a held-out test set.

### Evaluation

The evaluation workflow includes:

- Macro ROC-AUC
- Macro average precision
- Macro F1-score
- Per-finding ROC curves
- Per-finding precision-recall curves
- Training and validation history plots

## Limitations and Disclaimer

This is an educational and research project, **not a clinically validated medical device**. Predictions are model estimates and must not be used to diagnose conditions, guide treatment, or replace professional medical evaluation.

Performance can vary with image acquisition methods, dataset composition, disease prevalence, and patient populations. The reported metrics do not establish clinical effectiveness or suitability for real-world medical use.

## Future Improvements

- Improve per-finding performance and decision-threshold selection
- Evaluate probability calibration and robustness on external datasets
- Add automated tests and experiment tracking
- Investigate model interpretability methods
- Deploy the Streamlit application for a public demonstration
