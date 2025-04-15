# Real-Time-Sign-Language-Translator

## Running the Project

### Prerequisites

- Python 3.10  
- TensorFlow 2.10  
- OpenCV  
- NumPy  
- Matplotlib  
- MediaPipe (for hand tracking)  
- Optional: GPU for faster training and inference  

---

### GPU Acceleration (Optional)

To enable GPU support:

**Install CUDA Toolkit**

- Download [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) from NVIDIA's website (compatible with TensorFlow 2.10).

**Install cuDNN**

- Download [cuDNN 8.6.0](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 11.8 and copy its contents to the CUDA directory.

**Verify GPU Setup**

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Steps to Execute the Project

**1. Dataset Collection**

```bash
python datacollection.py
```

- Press `S` to capture images.
- Press `Q` to quit the script.

---

**2. Data Augmentation**

```bash
python dataAug.py
```

**3. Model Training**

Train any of the four models by running:

```bash
python model1.py   # ResNet50  
python model2.py   # MobileNetV3  
python model3.py   # CNN + ViT  
python model4.py   # CNN + LSTM  
```

Each script saves the model in the `Model/` directory.

---

**4. Real-Time Inference**

To test your trained model in real-time:

```bash
python test.py
```

At runtime, you'll be prompted to select a model:

Select a model to load:
- ResNet50
- MobileNetV3
- CNN + ViT
- CNN + LSTM

Enter your choice (1/2/3/4): 

The webcam feed will launch and display predictions.  
Press `Q` to quit the application.
