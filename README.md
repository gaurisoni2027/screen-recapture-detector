# 📷 Screen Recapture Detector

> A lightweight computer vision system that detects whether an image is a **real photograph** or a **photo of a digital screen (screen recapture)** using **EfficientNet-B0** and **PyTorch**.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)
![Computer Vision](https://img.shields.io/badge/Computer-Vision-success?style=for-the-badge)

</p>

---

## 🚀 Live Demo

🌐 **Web Application**

**<https://screen-recapture-detector-bygauri.streamlit.app/>**

🎥 **Video Demonstration**

**<https://drive.google.com/file/d/1VRdDA_t8VBR_yCALZIwxKYdKfvqfkAhp/view?usp=sharing>**

---

## 📸 Application Preview

<p align="center">
<img src="assets/homepage.png" width="900">
</p>

---

# 📖 Overview

Screen recapture attacks are a common challenge in identity verification and document authentication systems. Instead of submitting an original photograph, an attacker may simply display an image on another device and capture a photo of that screen.

This project presents a lightweight deep learning solution capable of distinguishing between:

- 📷 **Real Photograph**
- 🖥️ **Photo of a Screen (Screen Recapture)**

The solution is designed for **fast CPU inference**, **small model size**, and future **mobile deployment**, making it suitable for real-world edge applications.

---

# ✨ Features

- Binary image classification
- EfficientNet-B0 transfer learning
- Lightweight inference pipeline
- CPU-friendly deployment
- Streamlit web application
- Image upload support
- Camera capture support
- Screen probability score (0–1)
- Inference latency reporting
- Production-style command line interface

---

# 🧠 Model

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 |
| Framework | PyTorch |
| Learning Strategy | Transfer Learning |
| Task | Binary Classification |
| Classes | Real Photo / Screen Recapture |

---

# 📊 Performance

| Metric | Result |
|---------|--------|
| Validation Accuracy | **~90%** |
| Model | EfficientNet-B0 |
| Framework | PyTorch |
| Inference | CPU |
| Output | Screen Probability (0–1) |

The model achieves a strong balance between **accuracy**, **speed**, and **lightweight deployment**, making it suitable for resource-constrained environments.

---

# 🖼 Sample Predictions

<p align="center">

<img src="assets/output1.png" width="46%">
&nbsp;&nbsp;

<img src="assets/test2.png" width="46%">

</p>

<p align="center">

<img src="assets/webcam_output.png" width="46%">
&nbsp;&nbsp;

<img src="assets/webcam_test3.png" width="46%">

</p>

---

# 🏗️ Project Structure

```text
screen-recapture-detector/

├── assets/
│   ├── homepage.png
│   ├── output1.png
│   ├── test2.png
│   ├── webcam_output.png
│   └── webcam_test3.png
│
├── data/
├── models/
│   └── efficientnet_best.pth
│
├── src/
│   ├── dataset.py
│   ├── train_model.py
│   ├── inference.py
│   ├── features.py
│   └── utils.py
│
├── app.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/gaurisoni2027/screen-recapture-detector.git
```

Move into the project

```bash
cd screen-recapture-detector
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch the Streamlit application

```bash
streamlit run app.py
```

---

# 💻 Command Line Usage

Run inference on a single image:

```bash
python predict.py image.jpg
```

Example Output

```text
0.3872
```

### Output Interpretation

| Score | Meaning |
|--------|---------|
| **0.0** | Real Photo |
| **1.0** | Photo of a Screen |

---

# 🎯 Design Choices

The project was designed with practical deployment constraints in mind.

- EfficientNet-B0 provides an excellent balance between accuracy and computational efficiency.
- Transfer learning enables strong performance using a relatively small dataset.
- CPU inference eliminates the need for specialized hardware.
- The prediction interface follows the assignment specification by returning a single floating-point score between **0** and **1**.
- A Streamlit interface was included to demonstrate real-time usability.

---

# ⚠️ Limitations

- Dataset size is relatively small compared to production-scale datasets.
- Extremely realistic screen recaptures may still be challenging.
- Performance may degrade under severe glare, reflections, or unusual lighting conditions.
- Additional device diversity would further improve generalization.

---

# 🚀 Future Improvements

- EfficientNet-Lite / MobileNetV3
- TensorFlow Lite deployment
- ONNX Runtime
- INT8 Quantization
- Larger and more diverse dataset
- Continuous retraining using newly observed fraud patterns
- Multi-class fraud detection

---

# 🙏 Acknowledgement

This project was developed as part of the **SalesCode AI Computer Vision Take-Home Assignment**, focusing on building a lightweight and efficient computer vision solution for detecting screen recapture attacks.

---

# 👩‍💻 Author

**Gauri Soni**

Computer Science Undergraduate

GitHub:  
**https://github.com/gaurisoni2027**