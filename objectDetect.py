from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("yolov8x.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    while True:
        success, img = cap.read()
        if not success:
            break

        vehicle_count = 0
        person_count = 0

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = "person" if class_name == "person" else "vehicle"

                # Draw rectangle around detected object with different colors for vehicles and persons
                if label == 'person':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green color for persons
                    person_count += 1
                    label_color = (0, 255, 0)  # Green color for persons
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red color for vehicles
                    vehicle_count += 1
                    label_color = (0, 0, 255)  # Red color for vehicles

                # Draw label with black text color
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, label_color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # Create a single rectangle box with white background for counts
        count_box_height = 50
        cv2.rectangle(img, (10, 10), (frame_width // 2 - 420, 10 + count_box_height), (255, 255, 255), -1)

        # Display counts inside the white rectangle
        cv2.putText(img, f'Persons: {person_count}', (20, 45),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, f'Vehicles: {vehicle_count}', (300, 45),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        yield img

    cv2.destroyAllWindows()
