
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from huggingface_hub import hf_hub_download

import streamlit.components.v1 as components
import plotly.graph_objects as go

st.set_page_config(
    page_title="Brain Tumor MRI Analysis",
    page_icon="🧠",
    layout="wide",
)


st.markdown("""
<style>
/* ===== PURE BLACK MEDICAL AI THEME ===== */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #000000 !important;
    color: #F5F7FA !important;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080808 !important;
}

/* Headings */
h1, h2, h3 {
    color: #F8FAFC !important;
    letter-spacing: -0.4px;
}

/* Supporting text */
.stMarkdown p,
[data-testid="stCaptionContainer"] {
    color: #A7B0BE;
}

/* Subtle separators */
hr {
    border-color: #262626 !important;
}

/* MRI uploader */
[data-testid="stFileUploader"] {
    max-width: 560px;
    margin-top: 0.4rem;
}

[data-testid="stFileUploader"] section {
    border: 1px dashed #3B7777 !important;
    border-radius: 12px !important;
    background: #0A0F10 !important;
    padding: 0.5rem 0.75rem !important;
    box-shadow: none !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    border-radius: 8px;
    border: 1px solid #315E5E;
    padding: 0.3rem 0.8rem;
    font-size: 0.85rem;
}

/* Small upload text */
[data-testid="stFileUploader"] small {
    font-size: 0.75rem;
}

/* Rounded image corners */
[data-testid="stImage"] img {
    border-radius: 10px;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #252525;
    border-radius: 10px;
    overflow: hidden;
}

/* Charts and metric cards */
[data-testid="stMetric"] {
    background: #0A0A0A;
    border: 1px solid #252525;
    border-radius: 10px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #315E5E;
}

/* Remove excessive visual clutter */
[data-testid="stDecoration"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)





# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

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
        width=220,
    )



    return heatmap





# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🧠 Brain Tumor MRI Analysis")

st.caption(
    "Multi-model MRI classification • Research and educational prototype"
)

st.markdown(
    "Ensemble of Base CNN + ResNet50 + EfficientNetB0, "
    "with Grad-CAM explainability"
)

st.divider()







# --------------------------------------------------
# 3D ROTATING BRAIN
# --------------------------------------------------
html_path = os.path.join(os.path.dirname(__file__), "brain_animation.html")

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        brain_html = f.read()

    components.html(
        brain_html,
        height=440,
        scrolling=False,
    )
else:
    st.warning("3D brain animation file not found.")



# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("🔬 Upload MRI Scan")
st.caption("Upload a brain MRI image to begin analysis.")

uploaded_file = st.file_uploader(
    "Choose MRI image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
    label_visibility="collapsed",
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

       
        preview_col, info_col = st.columns([1, 1])

        with preview_col:
            st.image(
                image,
                caption="Uploaded MRI image",
                width=300,
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

            progress.empty()
            status.success("Analysis completed.")

            st.divider()
            st.subheader("📊 Prediction Summary")
            
            st.caption("Predicted class and model output probability")

            # Soft-voting ensemble prediction
            ensemble_index = int(np.argmax(ensemble_probabilities))
            ensemble_class = (
                CLASS_NAMES[ensemble_index]
                .replace("_", " ")
                .title()
            )
            ensemble_probability = ensemble_probabilities[ensemble_index] * 100

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #082b2b, #0A1115);
                    border: 1px solid #32CFC3;
                    border-radius: 14px;
                    padding: 20px 22px;
                    margin: 12px 0 20px 0;
                ">
                    <div style="
                        color: #55D6BE;
                        font-size: 13px;
                        font-weight: 700;
                        letter-spacing: 1px;
                    ">🧠 SOFT-VOTING ENSEMBLE PREDICTION</div>
                    <div style="
                        color: #F5F7FA;
                        font-size: 28px;
                        font-weight: 750;
                        margin: 10px 0;
                    ">{ensemble_class}</div>
                    <div style="color: #55D6BE; font-size: 17px; font-weight: 650;">
                        {ensemble_probability:.2f}% output probability
                    </div>
                    <div style="color: #A7B0BE; font-size: 12px; margin-top: 8px;">
                        Mean of the probability outputs from all three models
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


            summary_cols = st.columns(4)

            for column, (model_name, probabilities) in zip(
                summary_cols, all_probabilities.items()
            ):
                best_index = int(np.argmax(probabilities))
                predicted_class = (
                    CLASS_NAMES[best_index]
                    .replace("_", " ")
                    .title()
                )
                confidence = probabilities[best_index] * 100

                with column:
                    st.markdown(
                        f"""
                        <div style="
                            background: #0A0A0A;
                            border: 1px solid #2A2A2A;
                            border-radius: 12px;
                            padding: 16px 12px;
                            min-height: 125px;
                        ">
                            <div style="color:#A7B0BE;font-size:13px;">
                                {model_name}
                            </div>
                            <div style="
                                color:#F5F7FA;
                                font-size:19px;
                                font-weight:650;
                                margin:8px 0;
                            ">{predicted_class}</div>
                            <div style="
                                color:#55D6BE;
                                font-size:14px;
                                font-weight:600;
                            ">{confidence:.2f}% output probability</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
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
                    for model_name, probabilities in all_probabilities.items()
                },
                index=CLASS_NAMES,
            )

            model_accuracies = {
                "Custom CNN": "70.81%",
                "ResNet50": "91.56%",
                "EfficientNetB0": "92.94%",
            }

            # Professional, muted colours for the dark theme
            model_colors = {
                "Custom CNN": "#7899B8",
                "ResNet50": "#55B8A8",
                "EfficientNetB0": "#A69AD8",
            }

            fig = go.Figure()

            for model_name in chart_data.columns:
                probabilities = all_probabilities[model_name] * 100
                predicted_index = int(np.argmax(all_probabilities[model_name]))
                predicted_class = CLASS_NAMES[predicted_index].replace("_", " ").title()

                fig.add_trace(
                    go.Bar(
                        name=model_name,
                        x=CLASS_NAMES,
                        y=probabilities,
                        marker_color=model_colors[model_name],
                        customdata=[
                            [predicted_class, model_accuracies[model_name]]
                            for _ in CLASS_NAMES
                        ],
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "Class: %{x}<br>"
                            "Output probability: %{y:.2f}%<br>"
                            "Model's predicted class: %{customdata[0]}<br>"
                            "Test-set accuracy: %{customdata[1]}"
                            "<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=20, r=20, t=25, b=20),
                paper_bgcolor="#000000",
                plot_bgcolor="#0F1117",
                font=dict(color="#D7DEE8"),
                xaxis=dict(
                    title="Class",
                    showgrid=False,
                ),
                yaxis=dict(
                    title="Output probability (%)",
                    range=[0, 100],
                    dtick=20,
                    gridcolor="#292E38",
                    zerolinecolor="#292E38",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                hoverlabel=dict(
                    bgcolor="#17212B",
                    bordercolor="#55B8A8",
                    font=dict(color="#FFFFFF", size=13),
                ),
            )

            st.plotly_chart(fig, use_container_width=True)



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
                predicted_class = (
                    CLASS_NAMES[predicted_class_index]
                    .replace("_", " ")
                    .title()
                )

                with col:
                    st.markdown(f"**{model_name}**")

                    display_gradcam(
                        loaded_models[model_name],
                        image,
                        model_name,
                        predicted_class_index,
                    )

                    st.caption(f"Pred: {predicted_class.lower()}")

            st.divider()
            st.subheader("📊 Test-set accuracy")

            performance = pd.DataFrame({
                "Model": [
                    "Custom CNN",
                    "ResNet50",
                    "EfficientNetB0",
                    "Soft-voting ensemble",
                ],
                "Accuracy": [
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
                height=185,
            )

            st.caption(
                "Previously measured test-set accuracy; not a guarantee "
                "for individual MRI predictions."
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
