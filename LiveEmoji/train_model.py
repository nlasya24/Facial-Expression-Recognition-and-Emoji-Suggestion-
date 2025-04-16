# train_model.py
from utils import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

# Load and preprocess
X_train, y_train = load_dataset('DATASET/train', img_size=(64, 64))
X_test, y_test = load_dataset('DATASET/test', img_size=(64, 64))

# Normalize and reshape
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)

# One-hot encoding
y_train = to_categorical(y_train - 1, 7)
y_test = to_categorical(y_test - 1, 7)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Save model
model.save("expression_model.keras")
