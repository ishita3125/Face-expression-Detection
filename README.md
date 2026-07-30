# 😊 Face Expression Detection System

A real-time Facial Expression Detection System developed using **Python, TensorFlow, Keras, OpenCV, and CustomTkinter**. The system detects facial expressions from a webcam and classifies emotions using a Convolutional Neural Network (CNN) trained on the FER2013 dataset.

---

## 📌 Features

- Real-time face detection using OpenCV Haar Cascade
- Emotion classification using CNN
- Live webcam prediction
- User-friendly GUI built with CustomTkinter
- Trained on FER2013 dataset
- Lightweight and easy to use

---

## 🎯 Detected Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pillow (PIL)
- CustomTkinter
- Scikit-learn

---

## 📂 Project Structure

```
Face-Expression-Detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── emotion_model.h5
├── haarcascade_frontalface_default.xml
├── train_model.py
├── predict.py
├── gui.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Face-Expression-Detection-System.git
```

Move into the project folder

```bash
cd Face-Expression-Detection-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python gui.py
```

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Model | CNN |
| Dataset | FER2013 |
| Framework | TensorFlow/Keras |
| Image Size | 48×48 |
| Input | Grayscale Face Image |
| Output | 7 Emotion Classes |

---

## 📈 Future Improvements

- Face recognition support
- Emotion history graph
- Attendance with emotion
- Mobile deployment
- Better accuracy using EfficientNet or MobileNetV3
- Mask detection integration

---

## 📋 Requirements

```
tensorflow
keras
opencv-python
numpy
Pillow
customtkinter
scikit-learn
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Author

**Ishita Singh**

B.Tech Computer Science & Engineering (Artificial Intelligence)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute.
