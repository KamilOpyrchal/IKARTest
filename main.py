import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os 
import glob
current_dir = os.path.dirname(os.path.abspath(__file__))

template = os.path.join(current_dir,'videos', '*.mp4')
video_list = glob.glob(template)
base_options = python.BaseOptions(model_asset_path=os.path.join(current_dir, 'pose_landmarker_full.task'))

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

POSE_CONNECTIONS = [
    # Twarz
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Tułów
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Lewa ręka
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Prawa ręka
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Lewa noga
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Prawa noga
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]
for vid in video_list:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            
            timestamp_ms = int((frame_index * 1000) / fps)
            frame_index += 1 
                
            pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            
            if pose_result.pose_landmarks:
                        for pose_landmarks in pose_result.pose_landmarks:
                            h, w, _ = frame.shape
                            
                            landmarks_px = []
                            for landmark in pose_landmarks:
                                cx, cy = int(landmark.x * w), int(landmark.y * h)
                                landmarks_px.append((cx, cy))
                            
                            for connection in POSE_CONNECTIONS:
                                start_idx = connection[0] 
                                end_idx = connection[1]   
                                
                                start_point = landmarks_px[start_idx]
                                end_point = landmarks_px[end_idx]
                                
                                cv2.line(frame, start_point, end_point, (245, 66, 230), 2)

                            for pt in landmarks_px:
                                cv2.circle(frame, pt, 5, (245, 117, 66), -1)
                        
            cv2.imshow('MediaPipe Tasks API', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()