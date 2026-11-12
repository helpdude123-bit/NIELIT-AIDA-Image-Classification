import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

st.set_page_config(
    page_title="Bean Leaf Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "final_bean_leaf_disease_cnn.keras"

CLASS_NAMES = ["Angular Leaf Spot", "Bean Rust", "Healthy"]
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {MODEL_PATH}. "
            "Place final_bean_leaf_disease_cnn.keras in the models/ folder."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Match training: RGB, 224x224 bilinear resize, float32 scaled to [0, 1]."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


st.title("🌿 Bean Leaf Disease Classifier")
st.subheader("Deep Learning Based Bean Leaf Disease Classification Using CNN")
st.write(
    "Upload a bean leaf image to obtain an AI-based preliminary "
    "classification among three categories."
)
st.divider()

try:
    model = load_model()
    st.success("CNN model loaded successfully!")
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

st.header("1. Upload Bean Leaf Image")
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Uploaded Bean Leaf Image",
                use_container_width=True
            )

        with right:
            st.header("2. Disease Classification")
            if st.button("Predict Disease", type="primary", use_container_width=True):
                input_image = preprocess_image(image)
                predictions = model.predict(input_image, verbose=0)[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index] * 100)

                st.success("Prediction completed!")
                st.subheader("Predicted Class")
                st.markdown(f"### 🌱 {predicted_class}")
                st.metric("Prediction Confidence", f"{confidence:.2f}%")
                st.progress(confidence / 100.0)

                st.subheader("Class Probabilities")
                for name, probability in zip(CLASS_NAMES, predictions):
                    score = float(probability * 100)
                    st.write(f"**{name}: {score:.2f}%**")
                    st.progress(score / 100.0)

                if confidence < 60:
                    st.warning(
                        "The model prediction has relatively low confidence. "
                        "Consider expert verification."
                    )
    except Exception as error:
        st.error(f"Image processing or prediction failed: {error}")

st.divider()

with st.expander("About This Project"):
    st.write(
        """
        This project uses a Convolutional Neural Network (CNN) to classify
        bean leaf images into three categories:

        1. Angular Leaf Spot
        2. Bean Rust
        3. Healthy

        Dataset: Makerere AI Lab Beans Dataset
        Input: 224 x 224 RGB image
        Final model: Baseline CNN
        """
    )

st.info(
    "This application provides an AI-based preliminary classification, "
    "not a confirmed agricultural diagnosis. Consult an agricultural "
    "expert before taking treatment decisions."
)
st.caption("NIELIT AIDA Capstone Project | Bean Leaf Disease Classification")
