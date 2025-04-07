import cv2
import cvzone
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))

    # Run detection
    results = model(frame, stream=True)

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = classnames[class_id] if class_id < len(classnames) else 'Unknown'
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_name == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_name} ({conf}%)', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected!', [x1, y2 + 30], scale=2, thickness=2, colorR=(0, 0, 255))

    cv2.imshow('Fall Detection - Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
