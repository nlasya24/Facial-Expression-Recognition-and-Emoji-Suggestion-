import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Load Olivetti dataset
olivetti_faces = np.load('olivetti_faces.npy')  # shape: (400, 64, 64)
olivetti_labels = np.load('olivetti_faces_target.npy')  # shape: (400,)

# Load augmented dataset
aug_faces = np.load('augmented_faces.npy')       # shape: (N, 64, 64)
aug_labels = np.load('augmented_labels.npy')     # shape: (N,)

# Combine datasets
faces = np.concatenate((olivetti_faces, aug_faces), axis=0)
labels = np.concatenate((olivetti_labels, aug_labels), axis=0)

# Normalize and reshape
faces = faces.astype('float32') / 255.0
faces = faces.reshape(faces.shape[0], 64, 64, 1)

# One-hot encode labels
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Model architecture
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save
model.save('face_expression_model.h5')
print("✅ Training complete and model saved as face_expression_model.h5")
