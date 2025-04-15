import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

# Set folder path for saving images
folder = "Data\\Hello"

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Exiting...")
        break

    hands, img = detector.findHands(img)
    imgWhite = None  # Initialize imgWhite 

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Cropping is within image bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        # Check 
        if imgCrop.size == 0:
            print("Warning: Cropped image is empty. Skipping this frame.")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White background

        aspectRatio = h / w

        # Resize while maintaining aspect ratio
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Center horizontally
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Center vertically

        # Display images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        if imgWhite is not None:  # Check if imgWhite is defined
            counter += 1
            image_path = os.path.join(folder, f"Image_{time.time():.0f}.jpg")
            cv2.imwrite(image_path, imgWhite)
            print(f"Saved: {image_path} | Count: {counter}")
        else:
            print("Warning: No hand detected. Skipping save.")

    elif key == ord("q"):  # Press 'q' to exit
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
