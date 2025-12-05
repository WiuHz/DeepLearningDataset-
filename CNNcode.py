import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image_dataset_from_directory

image_dataset_from_directory = keras.preprocessing.image_dataset_from_directory

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

train_data = image_dataset_from_directory(
    "DetectionDataset/splittedDataset/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_data = image_dataset_from_directory(
    "DetectionDataset/splittedDataset/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_data = image_dataset_from_directory(
    "DetectionDataset/splittedDataset/val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)
    
num_classes = len(train_data.class_names)
print(f"Number of classes detected: {num_classes}")

input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

rescale = layers.Rescaling(1./255)

def preprocess_dataset(dataset):
    ds = dataset.map(lambda x, y: (rescale(x), y))
    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_data = preprocess_dataset(train_data)
test_data = preprocess_dataset(test_data)
val_data = preprocess_dataset(val_data)

print("Model Input Shape:", input_shape)



model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
        
        layers.MaxPooling2D(pool_size=(2,2)),
        
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 32
epochs = 10
history = model.fit(
    train_data, 
    validation_data=val_data,
    epochs=epochs
)
score_train = model.evaluate(train_data, verbose=0)
score_test = model.evaluate(test_data, verbose=0)

print(f"Train Loss: {score_train[0]*100}%")
print(f"Train Accuracy: {score_train[1]*100}%")
print(f"Test Loss: {score_test[0]*100}%")
print(f"Test Accuracy: {score_test[1]*100}%")