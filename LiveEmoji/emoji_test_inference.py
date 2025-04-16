import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("expression_model.keras")  
# Labels (adjust if different)
labels = ['1', '2', '3', '4', '5', '6', '7']

# Expression and emoji mapping
label_emoji_map = {
    '1': ('Angry', '😠'),
    '2': ('Disgust', '🤢'),
    '3': ('Fear', '😨'),
    '4': ('Happy', '😄'),
    '5': ('Sad', '😢'),
    '6': ('Surprise', '😲'),
    '7': ('Neutral', '😐')
}

def predict_expression_with_emoji(image_path):
    IMG_SIZE = (64, 64)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Could not read the image.")
        return

    img = cv2.resize(img, IMG_SIZE)
    img = img.reshape(1, 64, 64, 1).astype("float32") / 255.0

    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    expression, emoji = label_emoji_map[predicted_label]
    print(f"✅ Expression Detected: {expression} {emoji}")

    # Display original image with expression + emoji as title
    original = cv2.imread(image_path)
    cv2.imshow(f"{expression} {emoji}", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    test_image_path = r"C:\Users\LASYA\OneDrive - Vignan University\Desktop\EmojiSuggestion\liveEmoji-main\DATASET\train\7\train_09748_aligned.jpg"
    predict_expression_with_emoji(test_image_path)
