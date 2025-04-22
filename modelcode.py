from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# === CONFIGURATION ===
video_path = 'input_videos/video3.mp4'
output_dir = 'output_videos'
output_file = 'annotated_output_with_graphs3.mp4'
model_path = 'best.pt'

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Load model
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

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Font and color settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
color_map = {
    "Low": (0, 255, 0),
    "Medium": (0, 255, 255),
    "High": (0, 0, 255)
}

# For graph data
frame_times = []
people_counts = []
density_labels = []

frame_index = 0
print("🚀 Processing video and collecting data...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_index / fps
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.3)
    annotated_frame = results[0].plot()

    # Count people (assuming class 0 is person)
    person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)

    # Determine density level
    if person_count <= 10:
        density_label = "Low"
        label_text = "Low Population Density"
    elif 11 <= person_count <= 25:
        density_label = "Medium"
        label_text = "Medium Population Density"
    else:
        density_label = "High"
        label_text = "High Population Density"

    label_color = color_map[density_label]

    # Overlay text
    cv2.putText(annotated_frame, label_text, (30, 50), font, font_scale, label_color, thickness, cv2.LINE_AA)
    cv2.putText(annotated_frame, f'Count: {person_count}', (30, 100), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Save data
    frame_times.append(timestamp)
    people_counts.append(person_count)
    density_labels.append(density_label)

    # Save frame to output video
    out.write(annotated_frame)
    frame_index += 1

# === STEP 2: Generate Graphs ===

def normalize_color(bgr):
    """Convert OpenCV BGR to Matplotlib RGB normalized"""
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)

# Graph 1: Time vs People Count
plt.figure(figsize=(10, 4))
plt.plot(frame_times, people_counts, color='blue')
plt.title("Time vs People Count")
plt.xlabel("Time (seconds)")
plt.ylabel("People Count")
plt.grid(True)
plt.tight_layout()
graph1_path = os.path.join(output_dir, "time_vs_people3.png")
plt.savefig(graph1_path)
plt.close()

# Graph 2: Density Category Distribution
density_counts = Counter(density_labels)
colors_for_pie = [normalize_color(color_map[k]) for k in density_counts.keys()]

plt.figure(figsize=(6, 6))
plt.pie(
    density_counts.values(),
    labels=density_counts.keys(),
    autopct='%1.1f%%',
    colors=colors_for_pie
)
plt.title("Population Density Distribution")
plt.tight_layout()
graph2_path = os.path.join(output_dir, "density_distribution3.png")
plt.savefig(graph2_path)
plt.close()

# === STEP 3: Append Graphs to Video ===

graph1_img = cv2.imread(graph1_path)
graph2_img = cv2.imread(graph2_path)

# Resize to match video frame
graph1_img = cv2.resize(graph1_img, (width, height))
graph2_img = cv2.resize(graph2_img, (width, height))

print("🧩 Appending graphs to the video...")
for _ in range(int(fps * 3)):  # 3 seconds for each
    out.write(graph1_img)
for _ in range(int(fps * 3)):
    out.write(graph2_img)

# Cleanup
cap.release()
out.release()
print(f"✅ Final annotated video with graphs saved at:\n{output_path}")
