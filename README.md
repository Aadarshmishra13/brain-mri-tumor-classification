# 🧠 Brain Tumor MRI Classification Using Deep Learning

A comparative deep learning project for classifying brain MRI images into four categories using a custom Convolutional Neural Network (CNN), ResNet50, and EfficientNetB0. The project explores transfer learning, fine-tuning, model evaluation, learning-rate experiments, ensemble approaches, and Grad-CAM-based explainability.

**Best reported test accuracy: 92.94% with EfficientNetB0.**

> **Medical disclaimer:** This project is intended for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used for medical diagnosis, treatment decisions, or other clinical decision-making.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Methodology and Workflow](#-methodology-and-workflow)
- [Model Architectures](#-model-architectures)
- [Experimental Evaluation](#-experimental-evaluation)
- [Results and Model Comparison](#-results-and-model-comparison)
- [Grad-CAM Explainability](#-grad-cam-explainability)
- [Ensemble and Learning-Rate Experiments](#-ensemble-and-learning-rate-experiments)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Live Application](#-live-application)
- [Installation and Usage](#-installation-and-usage)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [Author](#-author)

---


## 📌 Project Overview

Brain tumor classification using MRI images is a computer vision task involving variations in tumor appearance and imaging characteristics.

This project evaluates deep learning models for classifying brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

A custom CNN trained from scratch is compared with pretrained ResNet50 and EfficientNetB0 architectures. The models are evaluated using test accuracy and additional performance metrics, with further experiments in transfer learning, ensemble methods, and Grad-CAM explainability.




## 🎯 Objectives

The main goals of this project are to:

- Build a baseline CNN for brain MRI classification.
- Explore transfer learning using **ResNet50** and **EfficientNetB0**.
- Compare model performance across different architectures.
- Analyze overall and class-specific performance using evaluation metrics.
- Investigate learning-rate effects on model performance.
- Experiment with model ensembles.
- Visualize model predictions using **Grad-CAM**.
- Develop an interactive **Streamlit application** for brain MRI classification.
- Discuss the limitations of deep learning in medical imaging.
---

## 🗂️ Dataset

This project uses the **Brain Tumor MRI Dataset** published on Kaggle by Masoud Nickparvar.

**Dataset source:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset contains four image classes:

| Class | Description |
|---|---|
| Glioma | MRI images categorized as glioma |
| Meningioma | MRI images categorized as meningioma |
| Pituitary | MRI images categorized as pituitary tumors |
| No Tumor | MRI images categorized as showing no tumor |

The dataset provides separate training and testing directories. The experiments use image preprocessing appropriate to the model being trained.

For the transfer-learning models, the documented input dimensions are **224 × 224 pixels**, with three-channel image inputs. The custom CNN uses grayscale MRI images according to the existing project methodology.

### Dataset considerations

- The dataset's class labels are used as the supervised learning targets.
- Training and testing data should remain separated during model development and evaluation.
- Preprocessing must be consistent between training and inference.
- Dataset-based results do not establish performance on external clinical datasets.

The dataset is intended here for machine-learning experimentation, not as a substitute for clinically curated and validated medical data.

---

## 🔬 Methodology and Workflow

The project follows a comparative experimental workflow, from image preparation to evaluation and model comparison.

```text
                 Brain Tumor MRI Dataset
                            |
                            v
                  Image Preprocessing
                            |
                            v
                  Model Development
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
          Custom CNN     ResNet50    EfficientNetB0
              |             |             |
              +-------------+-------------+
                            |
                            v
                  Model Evaluation
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
          Metrics      Confusion Matrix   ROC-AUC
                            |
                            v
                    Model Comparison
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Grad-CAM Analysis         Experimental Studies
                                 (Learning Rate /
                                     Ensembles)
                            |
                            v
                    Streamlit Inference App
```

### Workflow stages

1. **Data preparation:** Load the MRI dataset and prepare images for model input.
2. **Preprocessing:** Apply the input dimensions, channel representation, and preprocessing required by each architecture.
3. **Model development:** Train a custom CNN and adapt pretrained ResNet50 and EfficientNetB0 models.
4. **Evaluation:** Measure test performance and generate supporting metrics and visualizations.
5. **Comparison:** Examine differences between architectures, including model size and reported accuracy.
6. **Explainability:** Use Grad-CAM to inspect image regions associated with predictions.
7. **Deployment:** Provide an interactive interface for uploading an MRI image and viewing the model's prediction.

---

## 🏗️ Model Architectures

Three primary architectures are evaluated in this project.

### 1. Custom Convolutional Neural Network (CNN)

A convolutional neural network designed specifically for this project.

The baseline model consists of convolutional and pooling layers followed by fully connected layers for four-class classification.

It uses grayscale MRI images with pixel-value normalization.

### 2. ResNet50

A pretrained ResNet50 architecture adapted to the MRI classification task using transfer learning and fine-tuning.

The original ImageNet classification head is replaced with a custom classification head for the four target classes.

Input images use three-channel RGB representation and ResNet50-specific preprocessing.

### 3. EfficientNetB0

A pretrained EfficientNetB0 model adapted through transfer learning and fine-tuning. It achieved the highest reported test accuracy among the three models.

The documented model configuration includes:

- ImageNet-pretrained weights.
- Input dimensions of 224 × 224 × 3.
- A Global Average Pooling layer.
- Fully connected classification layers.
- Dropout regularization.
- A four-class softmax output.


### Why compare these architectures?

The custom CNN provides a baseline for learning from the selected dataset, while ResNet50 and EfficientNetB0 provide pretrained feature representations.

Comparing them helps investigate the trade-offs between predictive performance, model size, training strategy, and practical deployment.

---

## 📊 Experimental Evaluation

Model evaluation is based on multiple measures rather than accuracy alone.

### Evaluation metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified test images |
| Precision | Measures the proportion of predicted class positives that are correct |
| Recall | Measures the proportion of actual class positives correctly identified |
| F1-score | Combines precision and recall using their harmonic mean |
| Confusion matrix | Shows correct and incorrect predictions by class |
| ROC-AUC | Evaluates class discrimination using receiver operating characteristic analysis, where applicable |
| Learning curves | Show training and validation accuracy and loss over training |
| Model size | Indicates the approximate storage required for a saved model |

For multiclass classification, precision, recall, and F1-score should be interpreted together with the averaging method used, such as macro or weighted averaging.

### Evaluation artifacts

The repository includes saved results and visualizations to support further analysis:

- `results/figures/three_model_confusion_matrices.png`
- `results/figures/three_model_roc_curves.png`
- `results/figures/efficientnetb0_learning_curves.png`
- `results/figures/efficientnetb0_finetuning_curves.png`
- `results/figures/all_models_recovered_learning_curves.png`

The corresponding evaluation tables and experiment outputs are available in [`results/metrics/`](results/metrics/).

These artifacts can be used to investigate class-specific errors, training behavior, and differences between the evaluated architectures.

---

## 📈 Results and Model Comparison

The following table summarizes the reported test accuracies from the current experiments.

| Model | Reported test accuracy | Saved model size |
|---|---:|---:|
| Custom CNN | 70.81% | 1.31 MB |
| ResNet50 | 91.56% | 207.72 MB |
| EfficientNetB0 | **92.94%** | 41.47 MB |

*Model sizes are the previously measured sizes of the saved model artifacts. Actual deployment storage and memory requirements can vary.*

### Key observations

- **EfficientNetB0 achieved the highest reported test accuracy**, at 92.94%.
- ResNet50 also achieved substantially higher reported accuracy than the custom CNN in these experiments.
- EfficientNetB0's saved artifact is considerably smaller than the saved ResNet50 artifact in this project.
- The custom CNN has the smallest saved artifact, making it a useful lightweight baseline.
- Overall accuracy alone does not reveal which classes are most difficult to classify; confusion matrices and per-class metrics are necessary for a fuller comparison.

These results describe the evaluated dataset and experimental setup. They should not be interpreted as proof that one model will outperform another on independently collected MRI data.

### Detailed results

For more information, see:

- [`final_model_metrics_comparison.csv`](results/metrics/final_model_metrics_comparison.csv)
- [`model_comparison.csv`](results/metrics/model_comparison.csv)
- [`model_comparison_complete.csv`](results/metrics/model_comparison_complete.csv)
- [`efficientnetb0_complete_evaluation.csv`](results/metrics/efficientnetb0_complete_evaluation.csv)
- [`custom_cnn_complete_evaluation.csv`](results/metrics/custom_cnn_complete_evaluation.csv)
- [`efficientnetb0_roc_auc.csv`](results/metrics/efficientnetb0_roc_auc.csv)
- [`resnet50_roc_auc.csv`](results/metrics/resnet50_roc_auc.csv)
- [`custom_cnn_roc_auc.csv`](results/metrics/custom_cnn_roc_auc.csv)

---

## 🔍 Grad-CAM Explainability

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is a visualization technique that creates heatmaps to highlight image regions that influence a model's prediction.

In this project, Grad-CAM is used to investigate which regions of an MRI image contribute to a model's output.

### Conceptual workflow

```text
       Input MRI Image
              |
              v
        Trained Model
              |
              v
      Target Class Score
              |
              v
     Gradient Computation
              |
              v
  Convolutional Feature Maps
              |
              v
       Grad-CAM Heatmap
              |
              v
   Overlay on Original MRI
```

### What Grad-CAM can show

- Regions associated with a model's prediction.
- Differences in activation patterns across models or images.
- Potential reliance on irrelevant image features that deserve further investigation.


### Important limitations

Grad-CAM is an interpretability aid, not a proof of clinical reasoning. Highlighted regions do not establish that the model has identified a tumor correctly or is relying on medically meaningful features.

The results should be interpreted alongside quantitative evaluation and, where relevant, expert review.


### Grad-CAM Visualizations

The Grad-CAM images generated during this project are available in the [`results/gradcam/`](results/gradcam/) folder.

These examples include both correct and incorrect predictions, illustrating how Grad-CAM can help inspect model behavior and investigate potential limitations.
---


## 🤝 Ensemble and Learning-Rate Experiments

In addition to comparing individual architectures, the project includes experiments involving learning-rate configurations and model ensembles.

### Learning-rate experiments

Learning rates influence how model parameters are updated during optimization. An unsuitable learning rate can hinder convergence or lead to unstable training.

The saved experiment outputs support investigation of learning-rate configurations and their relationship to model performance.

### Ensemble experiments

An ensemble combines predictions from multiple models to make a final classification. In this project, a soft-voting ensemble was tested alongside the individual models.

The ensemble achieved 92.25% accuracy, while EfficientNetB0 performed best with 92.94% accuracy. Therefore, combining the models did not improve performance in this experiment.
Related files include:

- [`cnn_learning_rate_experiment.csv`](results/metrics/cnn_learning_rate_experiment.csv)
- [`learning_rate_configuration.csv`](results/metrics/learning_rate_configuration.csv)
- [`ensemble_comparison.csv`](results/metrics/ensemble_comparison.csv)

---

## 🧪 Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow, Keras |
| Architectures | CNN, ResNet50, EfficientNetB0 |
| Data processing | NumPy, Pandas |
| Evaluation | Scikit-learn |
| Image processing | Pillow, OpenCV |
| Visualization | Matplotlib, Seaborn |
| Development environment | Jupyter Notebook, Google Colab |
| Application | Streamlit |
| Version control | Git, GitHub |
| Model storage | Keras model format, Git LFS where configured |

---

## 📁 Repository Structure

The repository is organized to separate application code, experiments, documentation, and evaluation outputs.

```text
brain-mri-tumor-classification/
│
├── notebooks/
│   ├── README.md
│   └── 01_data_exploration.ipynb
│
├── results/
│   ├── figures/
│   │   ├── README.md
│   │   ├── all_models_recovered_learning_curves.png
│   │   ├── efficientnetb0_finetuning_curves.png
│   │   ├── efficientnetb0_learning_curves.png
│   │   ├── three_model_confusion_matrices.png
│   │   └── three_model_roc_curves.png
│   │
│   ├── metrics/
│   │   ├── README.md
│   │   ├── final_model_metrics_comparison.csv
│   │   ├── model_comparison.csv
│   │   ├── model_comparison_complete.csv
│   │   ├── efficientnetb0_complete_evaluation.csv
│   │   └── ...
│   │
│   └── predictions/
│       ├── README.md
│       └── custom_cnn_test_predictions.csv
│
├── reports/
│   ├── README.md
│   └── Brain_Tumor_MRI_Classification_Project_Report_Polished.docx
│
├── src/
│   └── README.md
│
├── app.py
├── fine_tuned_efficientnetb0.keras
├── requirements.txt
├── .gitattributes
└── README.md
```

*The tree highlights the main project files and representative artifacts; it omits some individual metrics files for readability. The `src/` directory currently serves as an organizational placeholder unless reusable source modules are added.*

---

## 🖥️ Live Application

The project includes a Streamlit application that allows users to upload an MRI image and view the trained EfficientNetB0 model's predicted class and confidence scores.

**[Open the live Brain Tumor MRI Classifier](https://brain-mri-tumor-classification-ntpgnptwbovxsfbnfyvup.streamlit.app)**


### Application capabilities

- Upload an MRI image through a web interface.
- Run inference using the configured EfficientNetB0 model.
- Display the predicted class.
- Present the model's output scores for the available classes.

The application is a demonstration of model inference. Prediction scores are model outputs and should not be interpreted as clinically calibrated probabilities or a medical diagnosis.

---

## ⚙️ Installation and Usage

### Prerequisites

- Python version compatible with the project's TensorFlow dependencies.
- Git.
- Sufficient memory and storage to load the trained model.
- The trained model artifact and required Python packages.

### 1. Clone the repository

```bash
git clone https://github.com/Aadarshmishra13/brain-mri-tumor-classification.git
cd brain-mri-tumor-classification
```

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser to interact with the application.

### Running the notebooks

Open `notebooks/01_data_exploration.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab to review the data-exploration workflow.

Training and evaluation experiments should be run using the corresponding project notebooks and code available in the development environment. The root notebook in this repository is the data-exploration notebook; it should not be mistaken for a complete collection of all model-training experiments.

---

## ⚠️ Limitations

### Dataset generalization

The reported results come from the selected MRI dataset. Performance may differ on data collected from other hospitals, scanners, populations, or imaging protocols.

### Dataset bias and leakage

A model can learn dataset-specific artifacts rather than robust disease-related features. Duplicate images, related images, or inappropriate splitting can also inflate measured performance. Patient-level separation and independent external evaluation are important where patient identifiers and suitable data are available.

### Class-specific errors

Overall accuracy can conceal false negatives and other important errors within individual classes. Per-class precision, recall, F1-score, confusion matrices, and suitable external validation are needed for a fuller assessment.

### Two-dimensional image classification

The project primarily treats MRI images as individual 2D images. This does not capture all the spatial and contextual information available in complete 3D MRI volumes.

### Explainability limitations

Grad-CAM visualizations can help investigate model behavior, but they do not prove that predictions are medically correct or that the model uses valid clinical reasoning.

### Clinical validation

The models have not been established as clinically validated diagnostic tools. The application is for educational demonstration and research only.

---

## 🚀 Future Improvements

Potential directions for extending this project include:

- Evaluating models on independent external datasets.
- Investigating patient-level data splitting and leakage prevention.
- Expanding systematic hyperparameter and learning-rate experiments.
- Performing more detailed per-class error analysis.
- Comparing additional pretrained architectures.
- Investigating model calibration and predictive uncertainty.
- Evaluating ensemble methods against the strongest individual model.
- Improving the reproducibility of training configurations and random seeds.
- Extending explainability analysis with additional methods and expert review.
- Exploring segmentation and 3D MRI analysis.
- Improving automated testing and deployment monitoring.

---

## 📚 References

1. **Dataset:** Masoud Nickparvar, *Brain Tumor MRI Dataset*. Kaggle.  
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2. **ResNet:** He, K., Zhang, X., Ren, S., and Sun, J. *Deep Residual Learning for Image Recognition.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.  
   https://doi.org/10.1109/CVPR.2016.90

3. **EfficientNet:** Tan, M., and Le, Q. V. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* Proceedings of the International Conference on Machine Learning (ICML), 2019.  
   https://proceedings.mlr.press/v97/tan19a.html

4. **Grad-CAM:** Selvaraju, R. R., et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.  
   https://doi.org/10.1109/ICCV.2017.74

---

## 👤 Author

**Aadarsh Mishra**

GitHub: [@Aadarshmishra13](https://github.com/Aadarshmishra13)

Project repository: [Brain MRI Tumor Classification](https://github.com/Aadarshmishra13/brain-mri-tumor-classification)

---

## 📜 License

This repository does not currently specify a license. If the project is intended for public reuse, an appropriate open-source license may be added.
