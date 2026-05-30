from ultralytics import YOLO
import os 
import glob


current_dir = os.path.dirname(os.path.abspath(__file__))

template = os.path.join(current_dir,'videos', '*.mp4')
video_list = glob.glob(template)

model = YOLO("yolo26m-pose.pt") 
for vid in video_list:
    results = model.track(vid, save=True, show=True, persist=True)
