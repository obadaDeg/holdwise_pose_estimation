import os
import shutil

# Define folder paths
folder = "raw_dataset"
video_folder = os.path.join(folder, "videos")
json_folder = os.path.join(folder, "json_data")

# Create output folder
output_folder = os.path.join(folder, "organized_data")
os.makedirs(output_folder, exist_ok=True)

# Get all files in video and JSON folders
video_files = sorted(os.listdir(video_folder))
json_files = sorted(os.listdir(json_folder))

# Function to extract the matching prefix (up to 18th character)
def extract_prefix(filename):
    return filename[:18]

# Determine the starting counter based on existing folders in output_folder
existing_folders = [int(f) for f in os.listdir(output_folder) if f.isdigit()]
counter = max(existing_folders, default=0) + 1

# Organize files into numbered folders
for json_file in json_files:
    json_prefix = extract_prefix(json_file)
    
    # Find matching video file
    matching_video = next((video for video in video_files if extract_prefix(video) == json_prefix), None)
    
    if matching_video:
        # Create a numbered folder
        numbered_folder = os.path.join(output_folder, str(counter))
        os.makedirs(numbered_folder, exist_ok=True)
        
        # Move JSON and video file to the numbered folder
        shutil.move(os.path.join(json_folder, json_file), os.path.join(numbered_folder, json_file))
        shutil.move(os.path.join(video_folder, matching_video), os.path.join(numbered_folder, matching_video))
        
        counter += 1

print(f"Files have been organized into {output_folder}.")
