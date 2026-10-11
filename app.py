
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

PROJECT_PATH = "/content/drive/MyDrive/Brain_Tumor_AI_Project"

MODEL_FILES = {
    "EfficientNetB0 (recommended)": "fine_tuned_efficientnetb0.keras",
    "ResNet50": "fine_tuned_resnet50.keras",
    "Custom CNN": "best_cnn.keras",
}

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

st.title("🧠 Brain Tumor MRI Classifier")
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
        st.image(image, caption="Uploaded MRI image", use_container_width=True)

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
                    # This CNN was trained on images scaled to [0, 1].
                    model_input = image_batch / 255.0
                elif selected_model == "ResNet50":
                    # Match ResNet50's training preprocessing.
                    model_input = (
                        tf.keras.applications.resnet50.preprocess_input(
                            image_batch.copy()
                        )
                    )
                else:
                    # EfficientNetB0 includes its own rescaling layer.
                    model_input = image_batch

                probabilities = model.predict(
                    model_input, verbose=0
                )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            st.subheader("Prediction")
            st.write(f"**Predicted class:** {predicted_class}")
            st.metric("Highest class probability", f"{confidence:.2f}%")

            st.caption(
                "These are model output probabilities, not calibrated "
                "clinical confidence estimates."
            )

            st.subheader("Output probabilities")
            for class_name, probability in zip(
                CLASS_NAMES, probabilities
            ):
                st.write(f"**{class_name}** — {probability * 100:.2f}%")
                st.progress(float(np.clip(probability, 0.0, 1.0)))

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()
st.caption(
    "Research/educational use only. Performance on uploaded images "
    "may differ from performance on the held-out dataset."
)
