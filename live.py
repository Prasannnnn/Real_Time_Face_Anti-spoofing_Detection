import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
import os

root_dir = os.getcwd()

# Ensure compatibility with the 'Functional' class issue in Keras
get_custom_objects()['Functional'] = tf.keras.Model

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier(os.path.join(root_dir, "models/haarcascade_frontalface_default.xml"))

# Load Anti-Spoofing Model Graph
with open(os.path.join(root_dir, 'antispoofing_models/antispoofing_model.json'), 'r') as json_file:
    loaded_model_json = json_file.read()

# Try to load the model from JSON and weights
try:
    # Load the model from JSON
    model = model_from_json(loaded_model_json)
    # Load weights for the model
    model.load_weights(os.path.join(root_dir, 'antispoofing_models/antispoofing_model.h5'))
    print("Model loaded successfully from JSON and weights.")
except Exception as e:
    print(f"Error loading model from JSON: {e}")
    print("Attempting to load model in the new format...")
    
    # If loading from JSON fails, load the model in the newer format
    model = load_model(os.path.join(root_dir, 'antispoofing_models/antispoofing_model.h5'))
    print("Model loaded successfully in the newer format.")

# Video capture from webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop and resize the face
            face = frame[max(0, y-5):min(y+h+5, frame.shape[0]), max(0, x-5):min(x+w+5, frame.shape[1])]
            resized_face = cv2.resize(face, (160, 160))

            # Normalize the image
            resized_face = resized_face.astype("float32") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)

            # Predict whether the face is real or spoofed
            preds = model.predict(resized_face)[0]

            # Classify based on model's output
            if preds > 0.5:
                label = 'spoof'
                color = (0, 0, 255)
            else:
                label = 'real'
                color = (0, 255, 0)

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    video.release()
    cv2.destroyAllWindows()
