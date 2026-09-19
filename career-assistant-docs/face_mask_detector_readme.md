# 😷 Face Mask Detector

A deep learning web app that detects whether a person is wearing a face mask, using Transfer Learning with MobileNetV2. Upload a photo or use your webcam — the app detects the face and classifies it as **Mask** or **No Mask** in real-time.

🔗 **Live App:** https://face-mask-detector-d5xy8dmxfbztn2qxgl9esa.streamlit.app/

## 🎯 What It Does

1. Detects faces in an uploaded photo or webcam snapshot using OpenCV's DNN face detector
2. Crops and classifies each detected face using a fine-tuned MobileNetV2 model
3. Displays the result with a labeled bounding box — green for Mask, red for No Mask, along with the model's confidence score

## 🧠 Tech Stack

- **Model:** MobileNetV2 (Transfer Learning) + custom classifier head
- **Face Detection:** OpenCV DNN (Caffe-based SSD face detector)
- **Framework:** TensorFlow / Keras
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud

## 📊 Results

- **Test Accuracy:** 99.9%
- Trained on the Face Mask 12K Images Dataset (10,000 training images, balanced classes)

## 🏗️ How It Works

The model uses MobileNetV2, pre-trained on ImageNet, with its base layers frozen. A custom classification head (GlobalAveragePooling → Dense → Dropout → Dense with sigmoid activation) was added on top and trained on the face mask dataset — allowing the model to reuse general visual knowledge while learning the specific mask/no-mask distinction quickly and with high accuracy.

For detection, OpenCV's DNN-based face detector is used instead of the traditional Haar Cascade method, since Haar Cascade struggles to detect faces that are partially covered by a mask.

## 🚀 Running Locally

```bash
git clone https://github.com/Gireesh08/face-mask-detector.git
cd face-mask-detector
pip install -r requirements.txt
streamlit run app.py
```

**Note:** Requires Python 3.11 (TensorFlow compatibility).

## 📁 Project Structure

```
face-mask-detector/
├── app.py                 # Streamlit app
├── face_mask_detector.h5  # Trained model
├── requirements.txt       # Dependencies
└── runtime.txt            # Python version pin
```

## 👤 Author

**Gireesh Adireddi**
[GitHub](https://github.com/Gireesh08) · [LinkedIn](https://www.linkedin.com/in/gireesh-adireddi-071362284/) · [Kaggle](https://www.kaggle.com/gireeshadireddi)
