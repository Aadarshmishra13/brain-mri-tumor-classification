
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1220, #101c2d, #0b1220);
        color: #e5edf7;
    }

    h1, h2, h3 {
        color: #f1f7ff;
        letter-spacing: -0.5px;
    }

    p, label {
        color: #c7d5e5;
    }

    div[data-testid="stFileUploader"] {
        background: #142438;
        border: 1px dashed #2dd4bf;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-baseweb="select"] > div {
        background-color: #142438;
        border-color: #2dd4bf;
        border-radius: 10px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #26384d;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_PATH = "."


MODEL_FILES = {
    "EfficientNetB0 (recommended)": "fine_tuned_efficientnetb0.keras",
}


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

st.title("🧠 Brain Tumor MRI Analysis")
st.caption("AI-assisted MRI classification • Research & educational prototype")
st.markdown("---")
st.write(
    "Upload an MRI image to see the model's predicted class "
    "and its output probabilities."
)
st.warning(
    "Educational prototype only. This model is not validated for "
    "clinical use and must not be used to diagnose or guide treatment."
)

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


st.markdown("### 🔬 Analyze an MRI scan")
st.write("Select a model, then upload an MRI image to view its prediction.")

selected_model = st.selectbox(
    "Choose a trained model",
    list(MODEL_FILES.keys())
)


uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.image(
                image,
                caption="Uploaded MRI image",
                use_container_width=True
            )

        with right_col:
            if st.button("Classify MRI", type="primary"):
                model_path = f"{PROJECT_PATH}/{MODEL_FILES[selected_model]}"

                with st.spinner("Loading model and classifying image..."):
                    model = load_model(model_path)

                    image_resized = image.resize((224, 224))
                    image_array = np.asarray(
                        image_resized, dtype=np.float32
                    )
                    image_batch = np.expand_dims(image_array, axis=0)

                    if selected_model == "Custom CNN":
                        model_input = image_batch / 255.0
                    elif selected_model == "ResNet50":
                        model_input = (
                            tf.keras.applications.resnet50.preprocess_input(
                                image_batch.copy()
                            )
                        )
                    else:
                        model_input = image_batch

                    probabilities = model.predict(
                        model_input, verbose=0
                    )[0]

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index]) * 100

                st.subheader("Prediction")
                st.write(f"**Predicted class:** {predicted_class}")
                st.metric(
                    "Highest class probability",
                    f"{confidence:.2f}%"
                )

                st.caption(
                    "These are model output probabilities, not calibrated "
                    "clinical confidence estimates."
                )

                st.subheader("Output probabilities")
                for class_name, probability in zip(
                    CLASS_NAMES, probabilities
                ):
                    st.write(
                        f"**{class_name}** — {probability * 100:.2f}%"
                    )
                    st.progress(
                        float(np.clip(probability, 0.0, 1.0))
                    )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()
st.caption(
    "Research/educational use only. Performance on uploaded images "
    "may differ from performance on the held-out dataset."
)
