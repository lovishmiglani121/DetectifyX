import math
from ultralytics import YOLO
import cv2
import os

def draw_bounding_boxes(file_path, output_path="static/processed"):
    model = YOLO('yolov8s.pt')
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "pen"]

    img = cv2.imread(file_path)
    results = model(img, stream=True)

    detected_class = None  # Store the class of the first detected object

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if detected_class is None:
                detected_class = class_name  # Store the class of the first detected object

            label = f'{class_name}{conf}'
            size_label = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            create = x1 + size_label[0], y1 - size_label[1] - 3
            cv2.rectangle(img, (x1, y1), create, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Generate the output file name based on the detected class and original file name
    output_file_name = os.path.join(output_path, f'{detected_class}_{os.path.basename(file_path)}')
    cv2.imwrite(output_file_name, img)

    return output_file_name

