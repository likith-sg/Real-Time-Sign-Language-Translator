import cv2
import os
import numpy as np
import random

# Input and output directory
INPUT_FOLDER = "Data\\Hello"
OUTPUT_FOLDER = "Data\\Hello_Augmented"
TARGET_SIZE = 1000  # Number of images per class

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Augmentation function
def augment_image(image):
    aug_images = []
    
    # Flip horizontally
    aug_images.append(cv2.flip(image, 1))
    
    # Rotate at multiple angles
    for angle in [10, -10, 20, -20, 30, -30, 45, -45]:
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        aug_images.append(rotated)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    aug_images.append(cv2.add(image, noise))
    
    # Adjust brightness & contrast with a wider range
    for _ in range(5):
        alpha = random.uniform(0.5, 1.5)  # Contrast
        beta = random.randint(-50, 50)    # Brightness
        aug_images.append(cv2.convertScaleAbs(image, alpha=alpha, beta=beta))
    
    # Apply Gaussian blur
    for k in [3, 5, 7]:  # Kernel sizes
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        aug_images.append(blurred)
    
    # Random cropping
    h, w, _ = image.shape
    for _ in range(3):
        x1, y1 = random.randint(0, w // 4), random.randint(0, h // 4)
        x2, y2 = random.randint(3 * w // 4, w), random.randint(3 * h // 4, h)
        cropped = cv2.resize(image[y1:y2, x1:x2], (w, h))
        aug_images.append(cropped)
    
    return aug_images

# Process images
images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png'))]
original_count = len(images)
print(f"Processing {INPUT_FOLDER}: Found {original_count} images.")

# Load all images
dataset = [cv2.imread(os.path.join(INPUT_FOLDER, img)) for img in images]
all_images = dataset.copy()

while len(all_images) < TARGET_SIZE:
    img = random.choice(dataset)
    augmented = augment_image(img)
    all_images.extend(augmented)

all_images = all_images[:TARGET_SIZE]  # Ensure exact count

# Save augmented images
for idx, img in enumerate(all_images):
    output_path = os.path.join(OUTPUT_FOLDER, f"Image_{idx}.jpg")
    cv2.imwrite(output_path, img)

print(f"Saved {len(all_images)} images in {OUTPUT_FOLDER}.")
print("Dataset augmentation completed!")