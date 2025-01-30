from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone

# Initialize the webcam and face detector
cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Offsets for adjusting bounding box size
offsetpercentagew = 10
offsetpercentageh = 20

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, img = cap.read()

    # If frame is not successfully captured, break the loop
    if not success:
        print("Failed to capture image from webcam")
        break

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']

            # Calculate offsets for width and height
            offsetw = (offsetpercentagew / 100) * w
            offseth = (offsetpercentageh / 100) * h

            # Adjust the bounding box with the offsets
            x = int(x - offsetw)
            w = int(w + offsetw * 2)
            y = int(y - offseth)
            h = int(h + offseth * 2)

            # Draw the adjusted bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Display the result
    cv2.imshow("Image", img)

    # Exit the loop if the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == "q":  # 27 is the ESC key
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
