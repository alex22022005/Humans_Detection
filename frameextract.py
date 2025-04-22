import cv2
import os

# Path to the input video
video_path = 'video9.mp4'  # Change this to your video file path
# Directory where frames will be saved
output_folder = 'Video9'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if there are no more frames

    # Save frame as image
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release video capture object
cap.release()
print(f"Saved {frame_count} frames to '{output_folder}' folder.")
