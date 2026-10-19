
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.dataset import LABELS, IMAGENET_MEAN, IMAGENET_STD
from src.model import create_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models" / "best_resnet18_extended.pth"
THRESHOLDS_PATH = ROOT / "models" / "thresholds.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_SIZE_MB = 10

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Chest X-ray AI Explorer",
    layout="wide",
)

st.title("Chest X-ray AI Explorer")



# --------------------------------------------------
# Load model and thresholds once per app session/cache
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


@st.cache_data
def load_thresholds():
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLDS_PATH}"
        )

    with open(THRESHOLDS_PATH, encoding="utf-8") as file:
        threshold_dict = json.load(file)

    missing = set(LABELS) - set(threshold_dict)
    if missing:
        raise ValueError(
            f"Missing thresholds for labels: {sorted(missing)}"
        )

    thresholds = np.array(
        [threshold_dict[label] for label in LABELS],
        dtype=np.float32,
    )

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Thresholds must all be finite.")

    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Thresholds must be between 0 and 1.")

    return thresholds


# --------------------------------------------------
# Inference
# --------------------------------------------------
def predict(image, model):
    # Match the preprocessing used for validation and testing.
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        if DEVICE.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)

        probabilities = torch.sigmoid(logits)[0].float().cpu().numpy()

    if probabilities.shape != (len(LABELS),):
        raise ValueError("Unexpected model output shape.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned invalid probabilities.")

    return probabilities


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "A multi-label image-classification project using a fine-tuned "
        "ResNet18 model."
    )
    st.metric("Output labels", len(LABELS))


# --------------------------------------------------
# Upload and preview
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Use a de-identified image for this research demo.",
)

if uploaded_file is None:
    st.info("Upload a PNG or JPEG image to get started.")
    st.stop()

if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
    st.error(f"Please upload an image smaller than {MAX_FILE_SIZE_MB} MB.")
    st.stop()

try:
    with Image.open(uploaded_file) as source:
        source.verify()

    uploaded_file.seek(0)

    with Image.open(uploaded_file) as source:
        image = source.convert("RGB")
        image.load()

except Exception:
    st.error("The uploaded file is not a valid supported image.")
    st.stop()


left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)
    st.caption(f"Image dimensions: {image.width} × {image.height}")

with right:
    st.subheader("Model output")
    run_inference = st.button(
        "Analyze image",
        type="primary",
        use_container_width=True,
    )

    if run_inference:
        try:
            with st.spinner("Running model inference..."):
                model = load_model()
                thresholds = load_thresholds()
                probabilities = predict(image, model)

            results = pd.DataFrame({
                "Condition label": LABELS,
                "Model score": probabilities,
                "Decision threshold": thresholds,
                "Above threshold": probabilities >= thresholds,
            })

            results = results.sort_values(
                "Model score", ascending=False
            ).reset_index(drop=True)

            st.session_state["prediction_results"] = results

        except Exception as exc:
            st.error(f"Inference failed: {exc}")


# --------------------------------------------------
# Results
# --------------------------------------------------
results = st.session_state.get("prediction_results")

if results is not None:
    st.divider()
    st.subheader("Prediction scores")



    display_results = results.copy()
    display_results["Model score"] = (
        display_results["Model score"] * 100
    ).map(lambda value: f"{value:.2f}%")

    display_results["Decision threshold"] = (
        display_results["Decision threshold"] * 100
    ).map(lambda value: f"{value:.2f}%")

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_data,
        file_name="chest_xray_model_scores.csv",
        mime="text/csv",
    )

    st.caption(
        "I built it for fun so guys, results should not be used for medical decisions (:"
    )
