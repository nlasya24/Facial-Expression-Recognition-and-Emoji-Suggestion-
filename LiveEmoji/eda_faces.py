import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
olivetti_images = np.load("olivetti_faces.npy")
olivetti_labels = np.load("olivetti_faces_target.npy")

aug_images = np.load("augmented_faces.npy")
aug_labels = np.load("augmented_labels.npy")

# EDA function
def explore_dataset(images, labels, title_prefix="Dataset"):
    print(f"\n===== {title_prefix} Stats =====")
    print(f"Total images: {images.shape[0]}")
    print(f"Image shape: {images.shape[1:]}")  # Should be (64, 64)
    print(f"Number of unique classes: {len(np.unique(labels))}")
    print(f"Samples per class: {np.bincount(labels)}")

    # Class distribution
    plt.figure(figsize=(10, 4))
    sns.countplot(x=labels)
    plt.title(f"{title_prefix} - Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Show sample images
    unique_classes = np.unique(labels)
    plt.figure(figsize=(15, 8))
    for i, cls in enumerate(unique_classes[:10]):  # Show first 10 classes
        idx = np.where(labels == cls)[0][0]  # First image of class
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Class {cls}")
        plt.axis('off')
    plt.suptitle(f"{title_prefix} - Sample Images from Classes")
    plt.tight_layout()
    plt.show()

    # Pixel distribution
    plt.figure(figsize=(6, 4))
    plt.hist(images.ravel(), bins=50, color='skyblue')
    plt.title(f"{title_prefix} - Pixel Intensity Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Perform EDA
explore_dataset(olivetti_images, olivetti_labels, title_prefix="Olivetti Faces")
explore_dataset(aug_images, aug_labels, title_prefix="Augmented Faces")
