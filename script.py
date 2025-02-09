import os
import shutil

# # Define folder paths
# folder = "raw_dataset"
# video_folder = os.path.join(folder, "videos")
# json_folder = os.path.join(folder, "json_data")

# # Create output folder
# output_folder = os.path.join(folder, "organized_data")
# os.makedirs(output_folder, exist_ok=True)

# # Get all files in video and JSON folders
# video_files = sorted(os.listdir(video_folder))
# json_files = sorted(os.listdir(json_folder))

# # Function to extract the matching prefix (up to 18th character)
# def extract_prefix(filename):
#     return filename[:18]

# # Determine the starting counter based on existing folders in output_folder
# existing_folders = [int(f) for f in os.listdir(output_folder) if f.isdigit()]
# counter = max(existing_folders, default=0) + 1

# # Organize files into numbered folders
# for json_file in json_files:
#     json_prefix = extract_prefix(json_file)
    
#     # Find matching video file
#     matching_video = next((video for video in video_files if extract_prefix(video) == json_prefix), None)
    
#     if matching_video:
#         # Create a numbered folder
#         numbered_folder = os.path.join(output_folder, str(counter))
#         os.makedirs(numbered_folder, exist_ok=True)
        
#         # Move JSON and video file to the numbered folder
#         shutil.move(os.path.join(json_folder, json_file), os.path.join(numbered_folder, json_file))
#         shutil.move(os.path.join(video_folder, matching_video), os.path.join(numbered_folder, matching_video))
        
#         counter += 1

# print(f"Files have been organized into {output_folder}.")
import os
import shutil
import random

# Define the input and output directories
folder = "./extracted_frames/workspace"
sub_folders = ["good_pose", "bad_pose", "cant_determine"]
output_folder = "./organized_data"
categories = ["train", "test"]
labels = ["good", "bad", "cant_determine"]

# Counters for numbering the output frames
counters = {"good": 0, "bad": 0, "cant_determine": 0}

# Create the output structure
for category in categories:
    for label in labels:
        os.makedirs(os.path.join(output_folder, category, label), exist_ok=True)

# Process files within each subfolder
for sub in sub_folders:
    sub_path = os.path.join(folder, sub)
    if not os.path.exists(sub_path):
        continue  # Skip if the subfolder doesn't exist

    # Determine label based on folder name
    if "good" in sub:
        label = "good"
    elif "bad" in sub:
        label = "bad"
    elif "cant_determine" in sub:
        label = "cant_determine"
    else:
        continue  # Skip folders not classified

    # Traverse all subdirectories and get files
    items = []
    for root, _, files in os.walk(sub_path):
        for file in files:
            if not file.endswith(".csv"):
                items.append(os.path.join(root, file))  # Full path of each file

    random.shuffle(items)  # Shuffle for random train/test split

    # Split items into train (80%) and test (20%)
    split_index = int(0.8 * len(items))
    train_files = items[:split_index]
    test_files = items[split_index:]

    # Copy and rename files into the corresponding output directories
    for category, files in zip(categories, [train_files, test_files]):
        for src_path in files:
            # Generate a new filename with the counter
            counters[label] += 1
            filename = os.path.basename(src_path)  # Extract the file name
            new_filename = f"{label}_{counters[label]:04d}{os.path.splitext(filename)[1]}"

            dest_path = os.path.join(output_folder, category, label, new_filename)
            try:
                shutil.copy(src_path, dest_path)
            except PermissionError as e:
                print(f"PermissionError: {e} - Skipping file: {src_path}")
            except Exception as e:
                print(f"Error: {e} - Skipping file: {src_path}")
