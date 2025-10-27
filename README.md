# 🥔 Potato Plant Disease Classification

This project aims to automatically detect and classify diseases in potato plants using deep learning. The model identifies whether a potato leaf is **Healthy**, affected by **Early Blight**, or affected by **Late Blight**. The goal is to assist farmers and agricultural researchers in early disease detection to prevent crop loss and improve yield quality.

---

## 📋 Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Preprocessing](#preprocessing)
* [Training](#training)
* [Evaluation](#evaluation)
* [Results](#results)
* [Technologies Used](#technologies-used)
* [How to Run](#how-to-run)
* [Future Enhancements](#future-enhancements)
* [References](#references)

---

## 🌾 Overview

Potato plants are highly vulnerable to fungal diseases such as *Early Blight* and *Late Blight*. Traditional identification methods are time-consuming and require expert knowledge.
This project automates the process using **Convolutional Neural Networks (CNN)** to classify images of potato leaves into:

* ✅ **Healthy**
* 🍂 **Early Blight**
* 🍃 **Late Blight**

---

## 📁 Dataset

* **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
* **Classes:**

  * `Potato___Healthy`
  * `Potato___Early_Blight`
  * `Potato___Late_Blight`
* **Image Count:** ~3,000–5,000 images
* **Image Size:** 256×256 pixels (resized)

---

## 🧠 Model Architecture

A **Convolutional Neural Network (CNN)** was used for classification.
Alternatively, **Transfer Learning** models such as **VGG16**, **ResNet50**, or **MobileNetV2** can also be used for improved accuracy.

**Example CNN architecture:**

```
Conv2D → ReLU → MaxPooling
Conv2D → ReLU → MaxPooling
Flatten → Dense(128) → ReLU
Dropout(0.5)
Dense(3) → Softmax
```

---

## 🧹 Preprocessing

* Image resizing to 256×256
* Normalization (pixel values / 255)
* Data augmentation:

  * Rotation, flipping, zoom, and brightness adjustment
* Train–validation–test split: 70% / 20% / 10%

---

## ⚙️ Training

* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam
* **Metrics:** Accuracy
* **Epochs:** 20–30
* **Batch Size:** 32

Model was trained using TensorFlow/Keras.

---

## 📊 Evaluation

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~98%  |
| Validation Accuracy | ~96%  |
| Test Accuracy       | ~95%  |

Confusion matrix and classification report were generated for deeper performance insights.

---

## 🖼️ Results

* The model successfully classifies healthy and diseased potato leaves.
* Helps in early detection of crop diseases, minimizing yield loss.
* Example prediction:

| Image                                                                                          | Predicted Label |
| ---------------------------------------------------------------------------------------------- | --------------- |
| ![Sample Image](https://raw.githubusercontent.com/example/potato-disease/main/sample_leaf.jpg) | Early Blight    |

---

## 🧰 Technologies Used

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
* **OpenCV** for image preprocessing
* **Google Colab / Jupyter Notebook** for development

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/potato-disease-classification.git
   cd potato-disease-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:

   ```bash
   python train_model.py
   ```
4. To test on new images:

   ```bash
   python predict.py --image path_to_image.jpg
   ```

---

## 🔮 Future Enhancements

* Develop a **mobile/web app** for real-time disease detection.
* Integrate **IoT sensors** for smart farming.
* Add **multi-crop support** for other plants like tomato, maize, etc.
* Deploy the model using **Flask**, **Streamlit**, or **FastAPI**.

---

## 📚 References

* [PlantVillage Dataset – Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
* TensorFlow Documentation
* Research papers on CNN-based crop disease detection

---

## 👩‍💻 Author

**Pretty**
🌸 *Data Science Enthusiast | Deep Learning Explorer | AI Innovator*

---

Would you like me to make this version slightly more **academic (for report submission)** or **GitHub portfolio-friendly (with emojis and clean formatting)**?
