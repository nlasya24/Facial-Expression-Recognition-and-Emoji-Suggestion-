# eda.py
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_dataset

X_train, y_train = load_dataset('DATASET/train')
X_test, y_test = load_dataset('DATASET/test')

print(f"Training shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test shape: {X_test.shape}, Labels: {y_test.shape}")

# Plot class distribution
sns.countplot(x=y_train)
plt.title("Class Distribution in Train Set")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# Show sample images
import random
plt.figure(figsize=(10, 5))
for i in range(10):
    idx = random.randint(0, len(X_train) - 1)
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(f"Label: {y_train[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
