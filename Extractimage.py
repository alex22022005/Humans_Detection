import os
import random
import shutil

# === CONFIGURATION ===
source_folder = 'Video5'
destination_folder = 'DataSet/final_images'
num_images = 50
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')  # Add/remove extensions as needed

# === CREATE DESTINATION FOLDER IF IT DOESN'T EXIST ===
os.makedirs(destination_folder, exist_ok=True)

# === LIST ALL IMAGE FILES IN SOURCE FOLDER ===
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(image_extensions)]

# === CHECK IF THERE ARE ENOUGH IMAGES ===
if len(all_images) < num_images:
    raise ValueError(f"Only found {len(all_images)} images, but {num_images} were requested.")

# === RANDOMLY PICK UNIQUE IMAGES ===
selected_images = random.sample(all_images, num_images)

# === COPY SELECTED IMAGES TO DESTINATION ===
for img in selected_images:
    src_path = os.path.join(source_folder, img)
    dst_path = os.path.join(destination_folder, img)
    shutil.copy2(src_path, dst_path)

print(f"{num_images} images have been copied to {destination_folder}")
