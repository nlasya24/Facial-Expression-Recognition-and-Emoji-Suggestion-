import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("expression_model.keras")
labels = ['1', '2', '3', '4', '5', '6', '7']  # update based on your training

cap = cv2.VideoCapture(0)
IMG_SIZE = (64, 64)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, IMG_SIZE)
    face = face.reshape(1, 64, 64, 1).astype("float32") / 255.0

    prediction = model.predict(face)
    pred_label = labels[np.argmax(prediction)]

    cv2.putText(frame, pred_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Live CNN Expression", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
