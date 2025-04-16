import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("expression_model.keras")
labels = ['1', '2', '3', '4', '5', '6', '7']  # or replace with actual emotion labels

def predict_expression(image_path):
    IMG_SIZE = (64, 64)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Could not read the image.")
        return

    img = cv2.resize(img, IMG_SIZE)
    img = img.reshape(1, 64, 64, 1).astype("float32") / 255.0

    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]
    print(f"✅ Predicted Expression: {predicted_label}")
    return predicted_label

# Example run
if __name__ == "__main__":
    predict_expression(r"C:\Users\LASYA\OneDrive - Vignan University\Desktop\EmojiSuggestion\liveEmoji-main\DATASET\test\7\test_2392_aligned.jpg")
