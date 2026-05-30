from ultralytics import YOLO


model = YOLO("yolo26n-pose.pt") 

results = model.track("trzy.mp4", save=True, show=True, persist=True)
