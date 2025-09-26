import torch
from ultralytics import YOLO
import cvzone
import math
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpuname = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpuname}")
else:
    print("Using CPU")

model = YOLO("../Yolo-Weights/yolov8l.pt").to(device)

classnames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "comb", "building"]

# 1. load image
img_path = "test.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not read image from {img_path}")
else:
    results = model(img, stream=False) # Use stream=False for single images
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

    # 2. Save the processed image to a new file
    output_path = "output.jpg"
    cv2.imwrite(output_path, img)
    print(f"Successfully saved processed image to {output_path}")

    # 3. Display the image in a window
    cv2.imshow("Processed Image", img)
    # Wait indefinitely for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()