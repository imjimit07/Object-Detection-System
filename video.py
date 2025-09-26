import torch
from ultralytics import YOLO
import time
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
              "teddy bear", "hair drier", "toothbrush", "comb"]

prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture("test.mp4")
# get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_source = int(cap.get(cv2.CAP_PROP_FPS))

# output
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 codec
out = cv2.VideoWriter(output_path, fourcc, fps_source, (frame_width, frame_height))

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Video has ended or failed to capture frame.")
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

    # FPS
    fps_display = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"Processing FPS: {fps_display:.2f}")
    out.write(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Successfully saved video to {output_path}")
cap.release()
out.release() # Release the VideoWriter
cv2.destroyAllWindows()