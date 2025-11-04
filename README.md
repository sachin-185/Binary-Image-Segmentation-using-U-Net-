# 🧠 Binary Image Segmentation using U-Net 
Developed by **SACHIN S**

---

## 📌 Task Overview  
Implement a **U-Net segmentation pipeline** for binary segmentation.  
Includes: data loader, augmentation, Dice-based loss, and CLI support for **train** + **predict** modes.

---

## 🛠️ Technology Stack

| Component | Tool / Library |
|-----------|----------------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras |
| Data Handling | NumPy, OpenCV, scikit-learn |
| Loss Functions | BCE + Dice |
| Evaluation | Dice Coefficient |
| Input Format | `data/images/*`, `data/masks/*` |
| Output | `unet_vehicle.h5`, `pred_mask.png`, `pred_prob.png` |

## Methodology
### Model Architecture (Why U-Net?)

A U-Net encoder–decoder architecture was selected because:

- Designed specifically for **pixel-level segmentation** tasks.  
- Uses **skip connections** to preserve spatial details lost during downsampling.  
- Performs well even with **limited datasets**, especially when combined with augmentation.  
- Efficient enough to train on mid-range GPUs or even a CPU for small datasets.  
- Provides smooth and accurate object boundaries, unlike plain CNN classifiers.


## 📂 Dataset

This project uses a **synthetic binary segmentation dataset** generated programmatically instead of real-world aerial images.

---

## 🧱 Architecture Overview:
- `Input: 256×256×3`
- `Encoder:`  
  ▸ `Conv2D(64) → ReLU → Conv2D(64) → ReLU → MaxPool`  
  ▸ `Conv2D(128) → ReLU → Conv2D(128) → ReLU → MaxPool`  
  ▸ `Conv2D(256) → ReLU → Conv2D(256) → ReLU → MaxPool`  
  ▸ `Conv2D(512) → ReLU → Conv2D(512) → ReLU → MaxPool`  
- `Bottleneck:`  
  ▸ `Conv2D(1024) → ReLU → Conv2D(1024) → ReLU`  
- `Decoder (Skip Connections):`  
  ▸ `UpSampling → Concat(c4) → Conv2D(512) ×2`  
  ▸ `UpSampling → Concat(c3) → Conv2D(256) ×2`  
  ▸ `UpSampling → Concat(c2) → Conv2D(128) ×2`  
  ▸ `UpSampling → Concat(c1) → Conv2D(64)  ×2`  
- `Output:`  
  ▸ `Conv2D(1, kernel=1, activation='sigmoid')`  
- **Loss:** `BCE + Dice Loss`  
- **Metric:** `Dice Coefficient`  
- **Optimizer:** `Adam (1e-4)`

✅ Skip connections  
✅ Fully convolutional  
✅ Outputs same resolution mask  

---

## ⚙️ Training Config

| Parameter | Value |
|-----------|--------|
| Image Size | 256×256 |
| Batch Size | 2 (default) |
| Epochs | 20 |
| Optimizer | Adam (1e-4) |
| Metric | Dice Coefficient |
| Train/Val Split | 80 / 20 |

---

### 📈 Training Summary (20 Epochs)

| Metric | Epoch 1 | Epoch 20 |
|--------|---------|----------|
| Training Loss | 1.5437 | 0.5038 |
| Validation Loss | 1.5645 | 0.1832 |
| Dice Coefficient (Train) | 0.1133 | 0.7226 |
| Dice Coefficient (Val) | 0.0486 | 0.8344 |

✅ Best model saved as: `unet_vehicle.h5`  

---

### 🔍 Raw Log (Condensed)

Epoch 1/20  - loss: 1.5437 - dice: 0.1133 - val_dice: 0.0486  
Epoch 10/20 - loss: 1.0492 - dice: 0.1487 - val_dice: 0.1297  
Epoch 15/20 - loss: 0.7974 - dice: 0.3760 - val_dice: 0.4309  
Epoch 20/20 - loss: 0.5038 - dice: 0.7226 - val_dice: 0.8344  

✅ Training complete   

--

## ✅ Pros of U-Net
- Works extremely well with limited training data (data-efficient).
- Skip connections help preserve spatial details → sharper masks.
- Symmetric encoder-decoder design enables multi-scale feature learning.
- Lightweight and trainable on mid-range GPUs/CPUs.
- Widely used and benchmark-proven in biomedical, satellite, and vehicle segmentation.
- Easy to modify (attention U-Net, ResU-Net, U-Net++, etc.).

## ❌ Cons of U-Net**
- Struggles with very large high-resolution images due to memory usage.
- No explicit handling of temporal or 3D context (not ideal for video or volumetric data unless extended).
- May overfit if the dataset is too synthetic or lacks real-world variance.
- Performance drops on extremely small or thin objects (boundaries may blur).
- Vanilla U-Net does not model long-range dependencies (no self-attention like Transformers).

--

## 🔍 How to Run

### 1️⃣ Train
```bash
python unet_train.py --data_dir data --epochs 20 --batch 4
```


### 2️⃣ Predict on a Single Image
```bash
python unet_train.py --img test.jpg --model unet_vehicle.h5
```

**Output files created:**
- `pred_prob.png` → grayscale probability heatmap
- `pred_mask.png` → binary mask (0/255)

---

## 📚 References
1. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
2. Chollet, F. (2015). *Keras: Deep Learning library for Theano and TensorFlow*.
3. Brownlee, J. (2017). *Deep Learning for Time Series Forecasting*. Machine Learning Mastery.
4. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
5. TensorFlow Documentation – https://www.tensorflow.org/api_docs
6. Scikit-learn Documentation – https://scikit-learn.org/
7. Time Series Forecasting Best Practices – Microsoft Research



