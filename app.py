import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0" 
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled" 
os.environ["DISPLAY"] = ""  

import cv2
import mediapipe as mp
from ultralytics import YOLO      
import time
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO('model/yolov8l.pt')

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

def is_focused(face_landmarks, image_width, image_height):
    left_eye_x = face_landmarks[159].x * image_width
    right_eye_x = face_landmarks[386].x * image_width
    gaze_direction = (left_eye_x + right_eye_x) / 2
    center_threshold = image_width * 0.4
    return center_threshold < gaze_direction < image_width - center_threshold

def calculate_centroid(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (box1_area + box2_area - inter_area)

st.title("📚 Classroom Attention Detection")
st.markdown("Monitor tingkat perhatian siswa realtime menggunakan deteksi wajah dan analisis fokus mata.")

uploaded_file = st.file_uploader("Upload video untuk analisis", type=["mp4", "avi", "mov"])
stframe = st.empty()
st_table = st.empty()
st_chart = st.empty()

if uploaded_file:
    tfile = f"temp_{uploaded_file.name}"
    with open(tfile, 'wb') as f:
        f.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detections = {}
    global_id_counter = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.resize(frame, (1280, 720))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results_yolo = model.predict(source=image_rgb, conf=0.5, save=False, show=False, imgsz=640)
        current_ids = []

        for box in results_yolo[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            if cls == 0:  # Kelas YOLO 'wajah'
                centroid = calculate_centroid(x1, y1, x2, y2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1] - 1, x2)
                y2 = min(image.shape[0] - 1, y2)

                matched_id = None
                for det_id, det_data in detections.items():
                    prev_centroid = det_data['centroid']
                    prev_box = det_data['bbox']

                    if euclidean_distance(centroid, prev_centroid) < 30 and iou([x1, y1, x2, y2], prev_box) > 0.5:
                        matched_id = det_id
                        break

                if matched_id is None:
                    matched_id = global_id_counter
                    global_id_counter += 1

                current_ids.append(matched_id)

                if matched_id not in detections:
                    detections[matched_id] = {
                        "centroid": centroid,
                        "bbox": [x1, y1, x2, y2],
                        "last_seen": time.time(),
                        "start_time": None,
                        "not_focused_start": None,
                        "duration_not_focused": 0,
                        "duration_focused": 0,
                        "status": "Unknown"
                    }

                detections[matched_id]['centroid'] = centroid
                detections[matched_id]['bbox'] = [x1, y1, x2, y2]
                detections[matched_id]['last_seen'] = time.time()

                face_roi = image_rgb[y1:y2, x1:x2]
                face_results = face_mesh.process(face_roi)

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        
                        if is_focused(face_landmarks.landmark, x2 - x1, y2 - y1):
                            status = "Focused"
                            color = (0, 255, 0)

                            if detections[matched_id]['not_focused_start']:
                                detections[matched_id]['not_focused_start'] = None

                            if detections[matched_id]['start_time'] is None:
                                detections[matched_id]['start_time'] = time.time()
                            else:
                                detections[matched_id]['duration_focused'] += time.time() - detections[matched_id]['start_time']
                                detections[matched_id]['start_time'] = time.time()
                        else:
                            status = "Not Focused"
                            color = (0, 0, 255)

                            if detections[matched_id]['not_focused_start'] is None:
                                detections[matched_id]['not_focused_start'] = time.time()
                            else:
                                detections[matched_id]['duration_not_focused'] += time.time() - detections[matched_id]['not_focused_start']
                                detections[matched_id]['not_focused_start'] = time.time()

                        detections[matched_id]['status'] = status

                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                        focused_time = round(detections[matched_id]['duration_focused'], 2)
                        not_focused_time = round(detections[matched_id]['duration_not_focused'], 2)

                        cv2.putText(image, f"ID {matched_id}: {status}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(image, f"Focused: {focused_time}s", (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.putText(image, f"Not Focused: {not_focused_time}s", (x1, y1 - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for det_id in list(detections.keys()):
            if det_id not in current_ids and time.time() - detections[det_id]['last_seen'] > 1:
                del detections[det_id]

        data = [
            {
                "ID": det_id,
                "Status": det_data['status'],
                "Focused Duration (s)": round(det_data['duration_focused'], 2),
                "Not Focused Duration (s)": round(det_data['duration_not_focused'], 2),
            }
            for det_id, det_data in detections.items()
        ]
        df = pd.DataFrame(data)
        st_table.dataframe(df)

        total_focused = sum(det['duration_focused'] for det in detections.values())
        total_not_focused = sum(det['duration_not_focused'] for det in detections.values())

        if total_focused + total_not_focused == 0:
            total_focused = 1
            total_not_focused = 0

        fig, ax = plt.subplots()
        ax.pie([total_focused, total_not_focused], labels=["Focused", "Not Focused"], autopct="%1.1f%%", colors=["green", "red"])
        ax.set_title("Focus Distribution")
        st_chart.pyplot(fig)

        stframe.image(image, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
