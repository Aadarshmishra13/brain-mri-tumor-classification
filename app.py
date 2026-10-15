
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Analysis",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #09111f, #111d30, #09111f);
    color: #e5edf7;
}
h1, h2, h3 { color: #f1f7ff; }
p, label { color: #c7d5e5; }
div[data-testid="stMetric"] {
    background: #142438;
    padding: 16px;
    border: 1px solid #26384d;
    border-radius: 12px;
}
div[data-testid="stFileUploader"] {
    background: #142438;
    border: 1px dashed #2dd4bf;
    border-radius: 14px;
    padding: 14px;
}
hr { border-color: #26384d; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
REPO_ID = "Aadarshmish/brain-tumor-mri-models"

MODEL_FILES = {
    "Custom CNN": "best_cnn.keras",
    "ResNet50": "fine_tuned_resnet50.keras",
    "EfficientNetB0": "fine_tuned_efficientnetb0.keras",
}

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILES[model_name],
    )
    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


def prepare_image(image, model_name):
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    if model_name == "Custom CNN":
        return batch / 255.0

    if model_name == "ResNet50":
        return tf.keras.applications.resnet50.preprocess_input(
            batch.copy()
        )

    # EfficientNetB0 includes rescaling in this trained model.
    return batch


def predict_probabilities(model, image, model_name):
    model_input = prepare_image(image, model_name)
    probabilities = np.asarray(
        model.predict(model_input, verbose=0)[0]
    ).reshape(-1)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} returned {len(probabilities)} outputs; "
            f"expected {len(CLASS_NAMES)}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{model_name} returned invalid probabilities.")

    return probabilities




# --------------------------------------------------
# GRAD-CAM HELPERS
# --------------------------------------------------

def make_gradcam_heatmap(model, image, model_name, class_index):
    """Generate Grad-CAM for the selected model."""
    try:
        image_batch = prepare_image(image, model_name)
        input_tensor = tf.convert_to_tensor(image_batch)

        # Find the last convolution layer.
        def find_conv(container):
            for layer in reversed(container.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return container, layer
                if isinstance(layer, tf.keras.Model):
                    result = find_conv(layer)
                    if result is not None:
                        return result
            return None

        found = find_conv(model)
        if found is None:
            st.error(f"No convolutional layer found for {model_name}.")
            return None

        container, conv_layer = found

        # --------------------------------------------------
        # Custom CNN: sequential gradient calculation
        # --------------------------------------------------
        if container is model:
            x = input_tensor
            activations = None

            with tf.GradientTape() as tape:
                for layer in model.layers:
                    x = layer(x, training=False)

                    if layer.name == conv_layer.name:
                        activations = x

                predictions = x
                score = predictions[:, class_index]

            if activations is None:
                st.error("Could not find the Custom CNN convolution output.")
                return None

            gradients = tape.gradient(score, activations)

            if gradients is None:
                st.error(f"No gradients available for {model_name}.")
                return None

            weights = tf.reduce_mean(gradients, axis=(1, 2))
            heatmap = tf.reduce_sum(
                activations[0] * weights[0][tf.newaxis, tf.newaxis, :],
                axis=-1,
            )

            heatmap = tf.maximum(heatmap, 0)
            heatmap /= (
                tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
            )

            return heatmap.numpy()

        # --------------------------------------------------
        # Transfer-learning models: ResNet50 / EfficientNetB0
        # --------------------------------------------------
        backbone = container

        backbone_cam = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[conv_layer.output, backbone.output],
        )

        with tf.GradientTape() as tape:
            activations, features = backbone_cam(
                input_tensor, training=False
            )

            x = features
            passed_backbone = False

            for layer in model.layers:
                if layer is backbone:
                    passed_backbone = True
                    continue

                if passed_backbone:
                    x = layer(x, training=False)

            score = x[:, class_index]

        gradients = tape.gradient(score, activations)

        if gradients is None:
            st.error(f"No gradients available for {model_name}.")
            return None

        weights = tf.reduce_mean(gradients, axis=(1, 2))
        heatmap = tf.reduce_sum(
            activations[0] * weights[0][tf.newaxis, tf.newaxis, :],
            axis=-1,
        )

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (
            tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
        )

        return heatmap.numpy()

    except Exception as error:
        st.error(f"Grad-CAM debug error ({model_name}): {error}")
        return None


def display_gradcam(model, image, model_name, class_index):
    heatmap = make_gradcam_heatmap(
        model, image, model_name, class_index
    )

    if heatmap is None:
        st.info(
            f"Grad-CAM could not be generated for {model_name}. "
            "The model prediction is still available above."
        )
        return None

    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    ).resize(image.size)

    base = np.asarray(
        image.convert("RGB"), dtype=np.float32
    )
    heat = np.asarray(
        heatmap_image, dtype=np.float32
    ) / 255.0

    colored = np.zeros_like(base)
    colored[..., 0] = 255 * heat
    colored[..., 1] = 180 * heat

    overlay = np.clip(
        0.55 * base + 0.45 * colored, 0, 255
    ).astype(np.uint8)

   
    st.image(
        overlay,
        use_container_width=True,
    )


    return heatmap


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🧠 Brain Tumor MRI Analysis")
st.caption(
    "Multi-model MRI classification • Research and educational prototype"
)

st.warning(
    "Educational use only. This tool is not clinically validated and "
    "must not be used to diagnose, exclude disease, or guide treatment. "
    "Predictions can be wrong, including on images from unseen sources."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Analysis settings")
    selected_cam_model = st.selectbox(
        "Grad-CAM model",
        list(MODEL_FILES.keys()),
        index=2,
    )
    st.caption(
        "The app downloads model weights from the public Hugging Face "
        "repository on first use. Downloads are cached by Hugging Face."
    )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.subheader("🔬 Upload an MRI image")
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        preview_col, info_col = st.columns([1, 1])

        with preview_col:
            st.image(
                image,
                caption="Uploaded MRI image",
                use_container_width=True,
            )

        with info_col:
            st.markdown("### Ready to analyze")
            st.write("Image dimensions: " + str(image.size))
            st.write("Models: 3 individual classifiers + soft voting")
            run_analysis = st.button(
                "🚀 Analyze MRI",
                type="primary",
                use_container_width=True,
            )

        if run_analysis:
            all_probabilities = {}
            loaded_models = {}

            progress = st.progress(0.0)
            status = st.empty()

            for index, model_name in enumerate(MODEL_FILES):
                status.write(f"Loading and running {model_name}...")
                model = load_model(model_name)
                loaded_models[model_name] = model

                all_probabilities[model_name] = predict_probabilities(
                    model, image, model_name
                )
                progress.progress((index + 1) / len(MODEL_FILES))

            # Soft voting: mean of the three probability vectors.
            ensemble_probabilities = np.mean(
                np.stack(list(all_probabilities.values())),
                axis=0,
            )
            all_probabilities["Soft-voting ensemble"] = (
                ensemble_probabilities
            )

            progress.empty()
            status.success("Analysis completed.")

            st.divider()
            st.subheader("📊 Prediction summary")

            summary_cols = st.columns(4)

            for column, (model_name, probabilities) in zip(
                summary_cols, all_probabilities.items()
            ):
                best_index = int(np.argmax(probabilities))
                with column:
                    st.metric(
                        model_name,
                        CLASS_NAMES[best_index].replace("_", " ").title(),
                        f"{probabilities[best_index] * 100:.2f}% output",
                    )

            st.caption(
                "Displayed percentages are model output probabilities, "
                "not calibrated clinical confidence estimates."
            )

            st.divider()
            st.subheader("📈 Class probability comparison")

            chart_data = pd.DataFrame(
                {
                    model_name: probabilities * 100
                    for model_name, probabilities
                    in all_probabilities.items()
                },
                index=CLASS_NAMES,
            )
            chart_data.index.name = "Class"

            st.bar_chart(chart_data, y_label="Output probability (%)")

            st.subheader("Detailed probabilities")
            detail_rows = []

            for model_name, probabilities in all_probabilities.items():
                for class_name, probability in zip(
                    CLASS_NAMES, probabilities
                ):
                    detail_rows.append({
                        "Model": model_name,
                        "Class": class_name,
                        "Output probability (%)": round(
                            float(probability * 100), 2
                        ),
                    })

            st.dataframe(
                pd.DataFrame(detail_rows),
                hide_index=True,
                use_container_width=True,
            )

            
            st.divider()
            st.subheader("🔥 Grad-CAM: Where Each Model Is Looking")

            cam_models = [
                "ResNet50",
                "EfficientNetB0",
                "Custom CNN",
            ]


            cam_cols = st.columns(3)

            for col, model_name in zip(cam_cols, cam_models):
                probabilities = all_probabilities[model_name]
                predicted_class_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[
                    predicted_class_index
                ].replace("_", " ").title()

                with col:
                    st.markdown(f"**{model_name}**")
                    st.caption(f"Prediction: {predicted_class}")

                    display_gradcam(
                        loaded_models[model_name],
                        image,
                        model_name,
                        predicted_class_index,
                    )


            st.divider()
            st.subheader("ℹ️ Model performance on the test set")

            performance = pd.DataFrame({
                "Model": [
                    "Custom CNN",
                    "ResNet50",
                    "EfficientNetB0",
                    "Soft-voting ensemble",
                ],
                "Held-out test accuracy": [
                    "70.81%",
                    "91.56%",
                    "92.94%",
                    "92.25%",
                ],
            })

            st.dataframe(
                performance,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "These are previously measured test-set accuracies, not "
                "a guarantee of performance on the uploaded image."
            )

    except Exception as error:
        st.error(
            "Analysis failed. Please check the uploaded image and "
            "deployment logs."
        )
        st.exception(error)

else:
    st.info("Upload an MRI image to begin the analysis.")

st.divider()
st.caption(
    "Research and educational prototype • Not for clinical use • "
    "Do not use these outputs to make medical decisions."
)

