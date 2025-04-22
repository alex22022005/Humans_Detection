from ultralytics import YOLO
import cv2
import os

# === CONFIG ===
video_path = 'input_videos/video1.mp4'  # path to input video
output_dir = 'output_videos'            # folder to save the output
output_file = 'annotated_output1.mp4'    # name of output video file
model_path = 'best1.pt'                  # your trained model

# === SETUP ===
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Load the YOLO model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    exit()

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === PROCESS FRAME BY FRAME ===
print("🚀 Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.3)

    # Annotate the frame with predictions
    annotated_frame = results[0].plot()

    # Write annotated frame to output
    out.write(annotated_frame)

# === CLEANUP ===
cap.release()
out.release()
print(f"✅ Output saved to: {output_path}")
