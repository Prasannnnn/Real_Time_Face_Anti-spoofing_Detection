import cv2
from ultralytics import YOLO

# Load the model
yolo = YOLO('yolov8n.pt')

# Load the video capture
videoCap = cv2.VideoCapture(0)

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    
    # Perform object detection (no tracking)
    results = yolo(frame)  # Use YOLO directly for detection

    for result in results:
        # Get the class names
        classes_names = result.names

        # Iterate over each box
        for box in result.boxes:
            # Check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # Get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # Convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the class
                cls = int(box.cls[0])

                # Get the class name
                class_name = classes_names[cls]

                # Get the respective color
                colour = getColours(cls)

                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # Put the class name and confidence on the image
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # Show the image
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()
