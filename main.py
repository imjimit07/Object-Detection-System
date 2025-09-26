import torch
from ultralytics import YOLO

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