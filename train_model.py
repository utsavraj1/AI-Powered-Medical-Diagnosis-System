import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset paths
TRAIN_PATH = r"C:\Users\utsha\Desktop\Medical Diagosis\Medical diagnosis using AI\Datasets\Blood_Cell_Cancer_image_data\train"
TEST_PATH = r"C:\Users\utsha\Desktop\Medical Diagosis\Medical diagnosis using AI\Datasets\Blood_Cell_Cancer_image_data\test"

# Load dataset
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = datagen.flow_from_directory(
    TRAIN_PATH, target_size=(224, 224), batch_size=32, class_mode="binary"
)

val_data = datagen.flow_from_directory(
    TEST_PATH, target_size=(224, 224), batch_size=32, class_mode="binary"
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save Model
model.save("Models/blood_cancer_model.h5")

print("✅ Model training complete! Saved as 'Models/blood_cancer_model.h5'")
