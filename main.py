import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('latestversion.pt')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or replace with the camera index

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to process each frame from the webcam
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO model on the frame
    results = model.predict(source=frame, show=False, conf=0.5)  # Set conf as needed (default 0.25)

    # Annotate the frame with detections
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO Detection", annotated_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
