import torch
from ultralytics import YOLO
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpuname = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpuname}")
else:
    print("Using CPU")

model = YOLO("../Yolo-Weights/yolov8l.pt").to(device)

classnames = ["person", "pen", "bottle", "phone", "headphones"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    results = model.predict(source=0, conf=0.5, device=device, show=True) # detect objects in webcam feed
    for r in results:
        print(r)
        for box in r.boxes:
            cls = int(box.cls[0])
            if classnames[cls] in [""person", "pen", "bottle", "phone", "headphones""]:
                print(f"Detected: {classnames[cls]} with confidence {box.conf[0]:.2f}")