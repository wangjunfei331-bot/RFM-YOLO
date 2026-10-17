
 
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-MSCU6.yaml")  # build a new model from scratch

 
# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
