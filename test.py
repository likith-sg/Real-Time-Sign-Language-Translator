import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import math
from collections import deque
import time
from transformers import TFViTModel, ViTFeatureExtractor

# Menu for model selection
print("\nSelect a model to load:")
print("1. ResNet50")
print("2. MobileNetV3")
print("3. CNN + ViT")
print("4. CNN + LSTM")

model_choice = input("Enter your choice (1/2/3/4): ").strip()

model_map = {
    "1": ("resnet50", "Model/resnet50_sign_language_model.keras"),
    "2": ("mobilenetv3", "Model/mobilenetv3_sign_language_model.keras"),
    "3": ("cnn_vit", "Model/cnn_vit_sign_language_model.keras"),
    "4": ("cnn_lstm", "Model/cnn_lstm_model.keras")
}

if model_choice not in model_map:
    print("Invalid choice. Exiting...")
    exit()

model_name, model_path = model_map[model_choice]

# Load Trained Model
print(f"\nLoading {model_name} model...")
try:
    start = time.time()

    if model_name == "cnn_vit":
        # Reconstruct feature extractor
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        # Define the preprocessing function used in the original model
        def vit_preprocess(x):
            mean = tf.constant(feature_extractor.image_mean)
            std = tf.constant(feature_extractor.image_std)
            return (x - mean) / std

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "TFViTModel": TFViTModel,
                "vit_preprocess": vit_preprocess
            }
        )
    else:
        model = tf.keras.models.load_model(model_path)

    print(f"{model_name} model loaded in {time.time() - start:.2f} seconds.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Labels (same for all models)
labels = ["Hello", "Yes", "No", "I Love You", "Okay", "Please", "Thank You"]

# Webcam and Hand Detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Scrolling Ticker Config
predicted_headlines = deque(maxlen=10)
scroll_offset = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2
scroll_speed = 2

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to read from camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            if w == 0 or h == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1 = max(y - offset, 0)
            y2 = min(y + h + offset, img.shape[0])
            x1 = max(x - offset, 0)
            x2 = min(x + w + offset, img.shape[1])

            imgCrop = img[y1:y2, x1:x2]
            aspectRatio = h / w

            if imgCrop.size == 0:
                continue  # Skip empty crop

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Prepare image for model
            input_img = cv2.resize(imgWhite, (224, 224))
            input_img = input_img.astype("float32") / 255.0
            prediction = model.predict(np.expand_dims(input_img, axis=0), verbose=0)
            index = np.argmax(prediction)
            label = labels[index]

            predicted_headlines.append(label)

            # Draw prediction
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 300, y - offset - 20), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

    except Exception as e:
        print(f"Prediction error: {e}")
        continue

    # Scrolling Ticker
    headline_text = "   >>>   " + " | ".join(predicted_headlines) + "   "
    text_size = cv2.getTextSize(headline_text, font, font_scale, font_thickness)[0]
    scroll_offset -= scroll_speed
    if scroll_offset < -text_size[0]:
        scroll_offset = imgOutput.shape[1]

    cv2.rectangle(imgOutput, (0, imgOutput.shape[0] - 40), (imgOutput.shape[1], imgOutput.shape[0]), (0, 0, 0), -1)
    cv2.putText(imgOutput, headline_text, (scroll_offset, imgOutput.shape[0] - 10),
                font, font_scale, (255, 255, 255), font_thickness)

    # Display output
    cv2.imshow("Real Time Sign Language Translator", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
