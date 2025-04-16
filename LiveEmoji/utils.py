import os
import cv2
import numpy as np

def load_dataset(path, img_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(path):
        class_dir = os.path.join(path, label)
        if not os.path.isdir(class_dir): continue
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(int(label))
    return np.array(images), np.array(labels)
