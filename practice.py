import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Streamlit Configuration
st.set_page_config(
    page_title="Real-Time Face Anti-Spoofing",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and Description
st.title("Real-Time Face Anti-Spoofing")
st.write("A professional based real-time face anti-spoofing and image quality assessment system.")

# Sidebar Configuration
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, step=0.05, help="Adjust confidence threshold for YOLO detection."
)
webcam_index = st.sidebar.number_input(
    "Webcam Index", min_value=0, value=0, step=1, help="Select the webcam index for the live feed."
)
show_bounding_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True, help="Display bounding boxes on the video feed.")
show_class_names = st.sidebar.checkbox("Show Class Names", value=True, help="Display class names with confidence.")

# Load YOLO Model
st.sidebar.write("Loading Anti Spoofing model...")
model = YOLO("latestversion.pt")
st.sidebar.success("Model loaded successfully!")

# Webcam Stream
st.write("### Live Webcam Feed")
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(webcam_index)

# Check if the webcam is opened
if not cap.isOpened():
    st.error("Error: Could not access the webcam. Please check your webcam index.")
    st.stop()

# Main Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture frame from webcam. Exiting...")
        break

    # Run YOLO on the frame
    results = model.predict(source=frame, conf=confidence_threshold, show=False)

    # Annotate frame with results
    if show_bounding_boxes or show_class_names:
        for box in results[0].boxes:
            # Bounding Box Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            # Draw bounding boxes
            if show_bounding_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Add Class Name and Confidence
            if show_class_names:
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

    # Display the processed frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Stop the stream on pressing "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
st.write("### Webcam feed stopped.")

