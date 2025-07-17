# Emotion Prediction from Facial Expressions with VGG19

This project builds a **multi-class image classification model** using a **VGG19** convolutional backbone to automatically predict human emotions from facial images. We train and evaluate on the **FER2013** dataset.

## 📊 Dataset Overview

The **FER2013** dataset contains **48×48** grayscale (converted to RGB) face images labeled with one of seven emotions:

| Label       | Emotion       |
| ----------- | ------------- |
| 0           | Anger         |
| 1           | Disgust       |
| 2           | Fear          |
| 3           | Happy         |
| 4           | Sad           |
| 5           | Surprise      |
| 6           | Neutral       |

- **Total images:** ~35,000  
- **Train/Validation/Test split:** 80% / 10% / 10%

## 🧹 Data Preprocessing

1. **Image resizing** to 48×48 pixels, converted from grayscale → RGB  
2. **Normalization**: pixel values scaled to [0, 1]  
3. **Label encoding** into one-hot vectors  
4. **Augmentation** via `ImageDataGenerator`:
   - Rotation (±15°)
   - Width/height shifts (±15%)
   - Horizontal flips  

## 📈 Exploratory Data Analysis

- **Class distribution** bar plot shows slight imbalance (Happy & Neutral most frequent).  
- **Sample grid** displays random faces per emotion.  

## 🧠 Model Building

- **Backbone:** `tf.keras.applications.VGG19`  
  - `include_top=False`, input shape=(48, 48, 3)  
  - Loaded pretrained “no-top” weights  
- **Head:**
  - `Flatten()`
  - `Dense(256, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(7, activation='softmax')`
- **Compile:**  
  ```python
  model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

**Callbacks:** EarlyStopping on `val_accuracy` (patience=10)

## 🧪 Training & Evaluation

* **Batch size:** 32
* **Epochs:** up to 50
* **Results:**

  * **Train accuracy:** \~75%
  * **Validation accuracy:** \~70%
* **Visualization:**

  * Training vs. validation **accuracy** and **loss** curves
  * **Confusion matrix** heatmap

## ⚙️ Installation

1. **Clone** this repository:

   ```bash
   git clone https://github.com/NishatTasnim01/Emotion-Prediction-Facial-Expressions.git
   cd Emotion-Prediction-Facial-Expressions
   ```
2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. **Launch** the notebook:

   ```bash
   jupyter notebook emotion-prediction-from-facial-expressions-final.ipynb
   ```

## 📁 Project Structure

```
├── emotion-prediction-from-facial-expressions-final.ipynb
├── models/
│   └── vgg19_emotion_weights.h5
├── data/
│   └── fer2013.csv
├── requirements.txt
└── README.md
```

## 👩‍💻 Developed By

🔗 [Nishat Tasnim](https://github.com/NishatTasnim01)

## 🙌 Acknowledgment

[FER2013 Dataset](https://datasets.activeloop.ai/docs/ml/datasets/fer2013-dataset/)


⭐ *If you found this project helpful, please give it a star!*
