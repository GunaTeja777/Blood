import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the trained model
model = load_model('mobilenetv2_lab_instruments.h5')

# Class names (must match your training folder names in order)
class_names = ['petri_dish', 'microscope_slide', 'beaker']  # 🧠 Replace with your actual folder names

# Image settings
IMG_SIZE = 224

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting live lab instrument detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for prediction
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_index = np.argmax(preds)
    confidence = np.max(preds)
    label = f"{class_names[class_index]} ({confidence * 100:.2f}%)"

    # Display prediction
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.imshow("Lab Instrument Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
