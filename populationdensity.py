from ultralytics import YOLO
import cv2
import os

# === CONFIG ===
video_path = 'input_videos/video3.mp4'  # path to input video
output_dir = 'output_videos'            # folder to save the output
output_file = 'annotated_output_with_density3.mp4'  # output video name
model_path = 'best.pt'                  # your trained model

# === SETUP ===
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Load YOLO model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    exit()

# Video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
color_map = {
    "Low": (0, 255, 0),     # Green
    "Medium": (0, 255, 255),# Yellow
    "High": (0, 0, 255)     # Red
}

# === PROCESS FRAME BY FRAME ===
print("🚀 Processing video with density labels...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.3)
    annotated_frame = results[0].plot()

    # Count number of people detected (class=0 usually means 'person')
    person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)

    # Determine density level
    if person_count <= 10:
        density_label = "Low Population Density"
        label_color = color_map["Low"]
    elif 11 <= person_count <= 25:
        density_label = "Medium Population Density"
        label_color = color_map["Medium"]
    else:
        density_label = "High Population Density"
        label_color = color_map["High"]

    # Add label text to the frame
    cv2.putText(annotated_frame, density_label, (30, 50), font, font_scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(annotated_frame, f'Count: {person_count}', (30, 100), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Write frame
    out.write(annotated_frame)

# === CLEANUP ===
cap.release()
out.release()
print(f"✅ Video with density labels saved to: {output_path}")
