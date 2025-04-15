# Real-Time Sign Language Translator

This project implements a Real-Time Sign Language Translator using deep learning techniques, including ResNet50, MobileNetV3, CNN + Vision Transformer (ViT), and CNN + LSTM. The system tracks hand gestures in real-time and translates them into text using TensorFlow, OpenCV, and MediaPipe.

## Features

- Real-time gesture recognition through webcam
- Multiple models for experimentation
- Data augmentation support
- Live inference with model selection

## Prerequisites

- Python 3.10+
- TensorFlow 2.10
- OpenCV
- NumPy
- Matplotlib
- MediaPipe
- (Optional) NVIDIA GPU with CUDA/cuDNN for acceleration

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/likith-sg/Real-Time-Sign-Language-Translator.git
cd Real-Time-Sign-Language-Translator
```

### 2. Set Up the Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**On Windows:**

```bash
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## GPU Acceleration (Optional)

### Install CUDA Toolkit

Download CUDA 11.2 from NVIDIA:  
https://developer.nvidia.com/cuda-11-2-2-download-archive

### Install cuDNN

Download cuDNN 8.1.0 for CUDA 11.2:  
https://developer.nvidia.com/rdp/cudnn-archive

Extract and copy the contents into your CUDA directory.

### Verify GPU Setup

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Running the Project

### 1. Dataset Collection

```bash
python datacollection.py
```

- Press `S` to save captured frames.
- Press `Q` to quit the script.

### 2. Data Augmentation

```bash
python dataAug.py
```

### 3. Train Models

Choose one of the following scripts to train:

```bash
python model1.py   # ResNet50
python model2.py   # MobileNetV3
python model3.py   # CNN + ViT
python model4.py   # CNN + LSTM
```

The trained model will be saved inside the `Model/` directory.

### 4. Real-Time Inference

To test your trained model in real time:

```bash
python test.py
```

You will be prompted to select a model:

```
Select a model to load:
1. ResNet50
2. MobileNetV3
3. CNN + ViT
4. CNN + LSTM

Enter your choice (1/2/3/4):
```

The webcam will launch and start showing real-time predictions.  
Press `Q` to quit the inference window.
