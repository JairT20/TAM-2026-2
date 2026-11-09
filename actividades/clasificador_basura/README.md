# Smart Waste Classification System (Computer Vision & Deep Learning)

## 1. Project Overview
An automated computer vision system designed to classify waste items into **Organic** and **Recyclable** categories using a Convolutional Neural Network (CNN) with Transfer Learning (MobileNetV2).

- **Domain:** Computer Vision / Smart Cities / Environmental Sustainability
- **Input:** Color RGB Images of waste materials ($224 \times 224 \times 3$)
- **Output:** Binary classification probability ($P(\text{Recyclable}) \in [0, 1]$)
- **Dataset:** Kaggle `Waste Classification data` (~25,000 images balanced between Organic and Recyclable)

---

## 2. Technical Architecture & Under-the-Hood Mechanics

```
Input Image (224x224x3)
        │
        ▼
┌───────────────────────────────────────────────┐
│ MobileNetV2 Feature Extractor (Pre-trained)   │
│ - Depthwise Separable Convolutions            │
│ - Trained on ImageNet (1.4M images, 1k classes)│
│ - Weights Frozen (Non-trainable base)         │
└───────────────────────────────────────────────┘
        │
        ▼ Feature Map (7x7x1280)
┌───────────────────────────────────────────────┐
│ Global Average Pooling 2D                     │
│ (Collapses spatial dimensions to 1x1280)      │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│ Dense Classification Head                     │
│ - Dense(128, activation='relu')               │
│ - Dropout(0.2) [Prevents Overfitting]         │
│ - Dense(1, activation='sigmoid')              │
└───────────────────────────────────────────────┘
        │
        ▼
Output: P(Recyclable)
```

---

## 3. Mathematical & Algorithmic Foundations

### A. Convolutional Operations (Feature Hierarchy)
1. **Low-level filters (early layers):** Detect edges, colors, sharp gradients, and simple geometric contrasts.
2. **Mid-level filters:** Assemble edges into textures, curves, and patterns (e.g., plastic bottle necks, apple stems, metallic reflections).
3. **High-level filters (deep layers):** Capture class-specific structural semantics (bottles, soda cans, banana peels, organic waste).

### B. Transfer Learning (Why MobileNetV2?)
- Instead of learning filter weights from random initialization (which requires huge datasets and weeks of GPU compute), we leverage **ImageNet priors**.
- **Depthwise Separable Convolutions:** Splits standard $3 \times 3$ convolution into a depthwise spatial filter and a $1 \times 1$ pointwise channel mixer, reducing computational cost by $\sim 8$ to $9\times$ with minimal loss in accuracy.

### C. Objective Function & Optimization
- **Loss Function:** Binary Cross-Entropy
  $$\mathcal{L}(y, \hat{y}) = - \left( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right)$$
- **Optimizer:** Adam (Adaptive Moment Estimation) with learning rate $\eta = 10^{-4}$.
- **Decision Threshold:** $\tau = 0.5$ (Values $\ge 0.5 \to$ Recyclable, $< 0.5 \to$ Organic).

---

## 4. Evaluation Metrics
- **Accuracy & F1-Score:** Overall classification reliability.
- **Confusion Matrix:** True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
- **ROC Curve & AUC:** Discriminative capacity across varying decision thresholds.

---

## 5. Next Steps for Implementation
1. Download dataset via Kaggle API (`kaggle datasets download -d techsash/waste-classification-data`).
2. Implement data augmentation pipeline (rotation, zoom, horizontal flips).
3. Build and compile Transfer Learning model using TensorFlow/Keras.
4. Train for 5–10 epochs in Google Colab using a free T4 GPU.
5. Create inference test script for real-world images (e.g., webcam or phone photo).
