import os
import json
import time
import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import keras_tuner as kt
import numpy as np

# Enable GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    device_details = tf.config.experimental.get_device_details(gpus[0])
    gpu_name = device_details.get("device_name", "Unknown GPU").replace("NVIDIA ", "NVIDIA ").title()
    print(f"Using {gpu_name}")
else:
    print("Using CPU")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Settings
DATASET_PATH = "Data"
MODEL_DIR = "Model"
PARAMS_PATH = os.path.join(MODEL_DIR, "best_params_mobilenetv3.json")
SELECTED_CLASSES = [
    "Hello_Augmented", "Yes_Augmented", "No_Augmented", "ILoveYou_Augmented",
    "Okay_Augmented", "Please_Augmented", "ThankYou_Augmented"
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
TRIALS = 3
VALIDATION_SPLIT = 0.3

# Timestamp
timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%Hhour-%Mmin-%Ssec")
TUNER_LOG_DIR = os.path.join("tuner_logs", f"mobilenetv3_{timestamp}")
os.makedirs(MODEL_DIR, exist_ok=True)

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)
train_generator = lambda: train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES, class_mode='categorical', subset='training'
)
val_generator = lambda: train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES, class_mode='categorical', subset='validation'
)

# Model Builder for Hyperparameter Tuning
def build_model(hp):
    input_tensor = Input(shape=(224, 224, 3), name='image')
    
    base_model = MobileNetV3Large(weights="imagenet", include_top=False, input_tensor=input_tensor)
    base_model.trainable = False  # Freeze pretrained layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1))(x)
    x = Dense(hp.Int("dense_units", 128, 512, step=128), activation='relu')(x)
    output = Dense(len(SELECTED_CLASSES), activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("lr", 1e-4, 3e-3, sampling="log")),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Hyperparameter Tuning
if os.path.exists(PARAMS_PATH):
    print("Best parameters already found. Loading model...")
    with open(PARAMS_PATH, 'r') as f:
        saved = json.load(f)
    best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "mobilenetv3_sign_language_model.keras"))
    total_time = saved.get("total_time_sec", 0)
else:
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=TRIALS,
        directory=TUNER_LOG_DIR,
        project_name='mobilenetv3_tuning'
    )

    start_time = time.time()
    tuner.search(
        train_generator(),
        epochs=EPOCHS,
        validation_data=val_generator(),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )
    total_time = time.time() - start_time

    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]

    with open(PARAMS_PATH, 'w') as f:
        json.dump({
            "learning_rate": best_hp.get('lr'),
            "dropout": best_hp.get('dropout'),
            "dense_units": best_hp.get('dense_units'),
            "timestamp": timestamp,
            "total_time_sec": total_time
        }, f, indent=2)

    best_model.save(os.path.join(MODEL_DIR, "mobilenetv3_sign_language_model.keras"))

# Evaluation
train_data = train_generator()
val_data = val_generator()

train_loss, train_acc = best_model.evaluate(train_data)
val_loss, val_acc = best_model.evaluate(val_data)

y_true = []
y_pred = []
val_data.reset()
for i in range(len(val_data)):
    x_batch, y_batch = val_data[i]
    preds = best_model.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
    if i >= val_data.samples // val_data.batch_size:
        break

report = classification_report(y_true, y_pred, target_names=SELECTED_CLASSES)

print(f"\nFinal Evaluation Metrics:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(report)

# Save training metrics
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics_mobilenetv3.json")
with open(METRICS_PATH, 'w') as f:
    json.dump({
        "total_training_time_seconds": total_time,
        "final_train_accuracy": train_acc,
        "final_val_accuracy": val_acc,
        "timestamp": timestamp
    }, f, indent=4)

print(f"\nModel saved at: {os.path.join(MODEL_DIR, 'mobilenetv3_sign_language_model.keras')}")
