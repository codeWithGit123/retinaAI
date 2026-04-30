# EfficientNetV2 with Attention for Diabetic Retinopathy Detection

## Overview

Diabetic Retinopathy (DR) is a leading cause of blindness worldwide, and early detection plays a crucial role in preventing severe vision loss. Automated deep learning models can assist ophthalmologists by providing fast and reliable screening from retinal fundus images.

This project presents a **transfer learning-based deep learning framework** for the **classification of Diabetic Retinopathy severity levels** using the **APTOS 2019 dataset**. The model integrates a **pretrained EfficientNetV2 architecture with CBAM attention**, class imbalance handling techniques, and interpretability using **Grad-CAM visualization**.

The system predicts **five DR grades** and provides both **classification metrics and visual explanations**, making it suitable for research and clinical decision support systems.

---

# Model Architecture

The proposed architecture combines **transfer learning, attention mechanisms, and advanced loss functions** to improve classification performance.

Pipeline:

Input Fundus Image
→ Image Preprocessing and Augmentation
→ EfficientNetV2-S (Transfer Learning Backbone)
→ CBAM Attention Module
→ Global Average Pooling
→ Fully Connected Classification Head
→ DR Grade Prediction (0–4)

### Key Components

**EfficientNetV2-S Backbone**

EfficientNetV2 is a modern convolutional neural network that provides an optimal balance between accuracy and computational efficiency. Pretrained weights from ImageNet are used to accelerate training and improve generalization.

**CBAM Attention Module**

The Convolutional Block Attention Module (CBAM) enhances feature learning by focusing on important spatial and channel-wise features within retinal images. This improves detection of lesions such as:

* Microaneurysms
* Hemorrhages
* Exudates

**Focal Loss**

APTOS2019 suffers from severe class imbalance. Focal Loss is used to penalize easy samples and focus more on minority classes.

**Weighted Sampling**

A WeightedRandomSampler ensures that underrepresented classes appear more frequently during training.

**Grad-CAM Explainability**

Grad-CAM visualizations highlight the retinal regions responsible for model predictions, improving interpretability and trust.

---

# Dataset

Dataset used: **APTOS 2019 Blindness Detection Dataset**

Source: Kaggle

Total Images: 3662

Class Distribution:

| Grade | Description      |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild DR          |
| 2     | Moderate DR      |
| 3     | Severe DR        |
| 4     | Proliferative DR |

Images are stored as retinal fundus photographs captured under different illumination conditions and camera settings.

Dataset structure used in this project:

train_1.csv
train_images/train_images/

---

# Data Preprocessing

The following preprocessing steps are applied:

* Image resizing to 380×380
* Normalization
* Horizontal and vertical flips
* Random rotations
* Brightness and contrast adjustments
* Shift-scale transformations

These augmentations improve model generalization and prevent overfitting.

---

# Training Configuration

Hardware: Google Colab T4 GPU (Free Tier)

Training Time: Approximately **25–30 minutes**

Key hyperparameters:

| Parameter     | Value            |
| ------------- | ---------------- |
| Batch Size    | 16               |
| Epochs        | 10               |
| Optimizer     | AdamW            |
| Learning Rate | 1e-4             |
| Scheduler     | Cosine Annealing |
| Loss Function | Focal Loss       |

Best model is automatically saved based on **Quadratic Weighted Kappa (QWK)**.

---

# Evaluation Metrics

To evaluate the performance of the model, the following metrics are used:

* Accuracy
* Precision
* Recall
* F1 Score
* Quadratic Weighted Kappa (QWK)
* Confusion Matrix

Quadratic Weighted Kappa is particularly important for DR grading tasks as it penalizes predictions that are far from the correct class.

---

# Grad-CAM Visualization

Grad-CAM is integrated to visualize the model’s attention regions.

This helps verify whether the model is focusing on clinically relevant retinal lesions.

The output visualization includes:

1. Original retinal image with predicted grade and confidence score
2. Heatmap showing important regions used by the model

---

# Example Output

Prediction Output:

Predicted Grade: 2
Confidence Score: 0.94

Visualization Output:

* Original Fundus Image with prediction label
* Grad-CAM heatmap highlighting pathological regions

---

# Project Features

✔ Transfer learning with EfficientNetV2
✔ Attention mechanism using CBAM
✔ Class imbalance handling
✔ Grad-CAM explainability
✔ Automatic model checkpointing
✔ Comprehensive evaluation metrics
✔ Single image testing pipeline
✔ Research-paper-ready architecture

---

# Repository Structure

project/

│
├── training_pipeline.ipynb
├── best_model.pth
├── README.md
│
├── dataset/
│   ├── train_1.csv
│   └── train_images/
│
└── outputs/
├── confusion_matrix.png
├── gradcam_example.png

---

# How to Run

1. Open the notebook in **Google Colab**
2. Install required libraries
3. Download dataset using KaggleHub
4. Run all training cells
5. Best model will be saved automatically
6. Run inference cell for testing images
7. Grad-CAM visualizations will be generated

---

# Download Trained Model

The trained model can be downloaded directly from Colab using:

```python
from google.colab import files
files.download("best_model.pth")
```

---

# Applications

This system can be used for:

* Automated DR screening
* Clinical decision support
* Tele-ophthalmology systems
* AI-based healthcare research

---

# Future Improvements

Future work may include:

* Vision Transformer architectures
* Multi-modal learning with patient metadata
* Ensemble deep learning models
* Lesion segmentation using U-Net
* Federated learning for medical data privacy

---

# Conclusion

This project demonstrates an efficient deep learning pipeline for diabetic retinopathy detection using retinal fundus images. By integrating transfer learning, attention mechanisms, and explainable AI techniques, the model achieves strong classification performance while maintaining interpretability.

The system shows potential for assisting ophthalmologists in early DR detection and reducing the burden of manual screening.

---

