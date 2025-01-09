#!/usr/bin/env python
# coding: utf-8

# To run this Application write this in your terminal:
# 
# ```py -3.10 -m venv myenv```
# 
# ```.\myenv\Scripts\Activate```
# 
# select the kernal to be myenv (Python 3.10.0)
# 
# ```py --version``` insure this version is 3.10.0
# 

# In[111]:


get_ipython().system('py --version')


# - **this cell could take some time to run**

# In[218]:


import subprocess
import importlib
import os
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor 

# Define the libraries and their pip install commands
libraries = {
    "tensorflow": "pip install tensorflow",
    "tensorflow-hub": "pip install tensorflow-hub",
    "numpy": "pip install numpy",
    "matplotlib": "pip install matplotlib",
    "pandas": "pip install pandas",
    "cv2": "pip install opencv-python", 
    "keras": "pip install keras",
    "pillow": "pip install pillow",
}

for library, command in libraries.items():
    try:
        importlib.import_module(library)
        print(f"{library} is already installed.")
    except ImportError:
        print(f"{library} not found. Installing...")
        try:
            subprocess.run(command.split(), check=True)
            print(f"{library} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {library}. Error: {e}")


import tensorflow as tf
import tensorflow_hub as hub
from keras import models, layers, applications, utils, preprocessing, callbacks, optimizers
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# In[113]:


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# In[207]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True) # Log device placement (on which device the operation is executed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0


# In[114]:


def video_to_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video at a specified frame rate.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save the extracted frames.
        frame_rate (int): Number of frames to save per second of video.

    Returns:
        None
    """
    
    if frame_rate <= 0:
        logging.error("Error: Frame rate must be a positive integer.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error(f"Error: Could not open video file '{video_path}'.")
        return

    # Get the video's frame rate and total number of frames
    video_fps = int(video.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        logging.error("Error: Invalid video FPS. Please check the video file.")
        video.release()
        return

    if frame_rate > video_fps:
        logging.warning(f"Warning: Frame rate ({frame_rate}) exceeds video FPS ({video_fps}). Adjusting to match video FPS.")
        frame_rate = video_fps

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Video FPS: {video_fps}, Total Frames: {frame_count}")

    # Calculate the frame interval for the desired frame rate
    frame_interval = video_fps // frame_rate

    # Extract and save frames
    frame_idx = 0
    saved_frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video

        # Save the frame at the specified interval
        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_idx += 1

        frame_idx += 1

    # Release the video capture object
    video.release()

    if saved_frame_idx == 0:
        logging.warning("No frames were saved. Adjust the frame rate and try again.")
    else:
        logging.info(f"{saved_frame_idx} frames saved to '{output_folder}'.")
        
    return saved_frame_idx


# In[115]:


target_extraction_folders = [
    "workspace/good_pose",
    "workspace/bad_pose",
    "workspace/cant_determine"
]


def process_session(session_path, label, output_dir, frame_rate=1):
    """
    Processes a single session folder, extracting frames and mapping to sensor data.
    """
    video_path, json_path = None, None

    for file_name in os.listdir(session_path):
        file_path = os.path.join(session_path, file_name)
        lower = file_name.lower()
        if lower.endswith('.mp4'):
            video_path = file_path
        elif lower.endswith('.json'):
            json_path = file_path

    if not video_path or not os.path.exists(video_path):
        logging.warning(f"No valid video found in: {session_path}")
        return
    if not json_path or not os.path.exists(json_path):
        logging.warning(f"No valid JSON file found in: {session_path}")
        return

    with open(json_path, 'r') as f:
        sensor_data = json.load(f)

    gyro_list = sensor_data.get("gyroscopeData", [])
    accel_list = sensor_data.get("accelerometerData", [])

    session_name = Path(session_path).name
    session_output = os.path.join(output_dir, label, session_name)
    os.makedirs(session_output, exist_ok=True)

    saved_frames = video_to_frames(video_path, session_output, frame_rate)
    if saved_frames == 0:
        logging.warning("No frames were extracted; skipping sensor mapping.")
        return

    if not gyro_list or not accel_list:
        logging.warning("Missing sensor data in JSON; skipping sensor mapping.")
        return

    frame_files = sorted([f for f in os.listdir(session_output) if f.lower().endswith('.jpg')])
    csv_path = os.path.join(session_output, "frame_sensor_mapping.csv")

    with open(csv_path, 'w') as csv_file:
        csv_file.write("frame_file,frame_idx,sensor_idx,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n")
        M = saved_frames
        N = len(gyro_list)
        L = len(accel_list)

        for i, frame_file in enumerate(frame_files):
            gyro_idx = int((i / M) * N)
            accel_idx = int((i / M) * L)

            gyro_data = gyro_list[gyro_idx]
            accel_data = accel_list[accel_idx]

            csv_file.write(
                f"{frame_file},{i},{gyro_idx},"
                f"{gyro_data.get('x', 0)},{gyro_data.get('y', 0)},{gyro_data.get('z', 0)},"
                f"{accel_data.get('x', 0)},{accel_data.get('y', 0)},{accel_data.get('z', 0)}\n"
            )

            # old_path = os.path.join(session_output, frame_file)
            # base, ext = os.path.splitext(frame_file)
            # new_filename = f"{base}_sensor_{gyro_idx}{ext}"
            # new_path = os.path.join(session_output, new_filename)
            # os.rename(old_path, new_path)

    logging.info(f"Frame-to-sensor mapping complete for: {session_path}")


# In[116]:


output_dir = "extracted_frames"
# frame_rate = 30
frame_rate = 1 # for demonstration purposes

def process_label(label):
        label_path = Path(label)
        if not label_path.exists():
            logging.warning(f"Label folder not found: {label_path}")
            return

        for session_name in os.listdir(label_path):
            session_path = os.path.join(label_path, session_name)
            if not os.path.isdir(session_path):
                continue

            try:
                process_session(session_path, label, output_dir, frame_rate)
            except Exception as e:
                logging.error(f"Error processing session '{session_path}': {e}")

with ThreadPoolExecutor() as executor:
    executor.map(process_label, target_extraction_folders)


# In[117]:


import pandas as pd
mapped_df = pd.read_csv("extracted_frames/workspace/good_pose/1/frame_sensor_mapping.csv")
mapped_df.head()


# In[118]:


IMAGE_SIZE = (224, 224)
DATA_DIR = "extracted_frames/workspace"
DATA_COLS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]


# In[193]:


# from keras import layers
def augment_image(img):
    """
    Apply data augmentation to a single image.

    Args:
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Augmented image as a NumPy array.
    """
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tensor = tf.image.random_flip_left_right(img_tensor)  # Random horizontal flip
    img_tensor = tf.image.random_brightness(
        img_tensor, max_delta=0.3
    )  # Random brightness
    img_tensor = tf.image.random_contrast(
        img_tensor, lower=0.8, upper=1.2
    )  # Random contrast
    img_tensor = tf.image.resize_with_crop_or_pad(
        img_tensor, IMAGE_SIZE[0] + 10, IMAGE_SIZE[1] + 10
    )  # Add padding
    img_tensor = tf.image.random_crop(
        img_tensor, size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )  # Random crop
    img_tensor = tf.image.random_saturation(
        img_tensor, lower=0.5, upper=1.5
    )  # playing with the saturation

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    img_tensor = tf.clip_by_value(img_tensor, 0.0, 255.0)  # Clip values
    return img_tensor.numpy()


# In[189]:


def load_data(data_dir, min_frames_per_session=1, augmentations_per_image=3):
    """
    Loads images, sensor data, and labels from the dataset directory, with data augmentation.

    Args:
        data_dir (str): Path to the dataset directory.
        min_frames_per_session (int): Minimum number of frames required to process a session.
        augmentations_per_image (int): Number of augmented images to generate per original image.

    Returns:
        Tuple: (images, sensors, labels)
    """
    images = []
    sensors = []
    labels = []

    for label_folder in ["good_pose", "bad_pose"]:
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.exists(label_path):
            logging.warning(f"Label folder not found: {label_path}")
            continue

        # Loop through each session folder
        session_folders = [f for f in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, f))]
        for session in session_folders:
            session_path = os.path.join(label_path, session)

            # Find the sensor mapping CSV
            csv_file = os.path.join(session_path, "frame_sensor_mapping.csv")
            if not os.path.exists(csv_file):
                logging.warning(f"Sensor mapping CSV not found: {csv_file}")
                continue

            # Load the sensor data from CSV
            df = pd.read_csv(csv_file)
            session_images = []
            session_sensors = []
            session_labels = []

            # Get all available frame filenames in the session directory
            available_frames = {os.path.basename(f) for f in os.listdir(session_path) if f.endswith('.jpg')}
            logging.debug(f"Available frames in {session_path}: {available_frames}")

            for _, row in df.iterrows():
                frame_file_name = row["frame_file"]
                if frame_file_name not in available_frames:
                    logging.warning(f"Frame file not found: {frame_file_name}")
                    continue

                frame_file_path = os.path.join(session_path, frame_file_name)
                try:
                    # Load and preprocess image
                    img = preprocessing.image.load_img(frame_file_path, target_size=IMAGE_SIZE)
                    img = preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]

                    # Apply augmentations
                    augmented_imgs = [img] + [augment_image(img) for _ in range(augmentations_per_image)]

                    # Load sensor data
                    sensor_features = np.array([
                        row["gyro_x"], row["gyro_y"], row["gyro_z"],
                        row["accel_x"], row["accel_y"], row["accel_z"]
                    ])

                    # Append data
                    for augmented_img in augmented_imgs:
                        session_images.append(augmented_img)
                        session_sensors.append(sensor_features)
                        session_labels.append(1 if label_folder == "good_pose" else 0)
                except Exception as e:
                    logging.error(f"Error processing frame {frame_file_path}: {e}")
                    continue

            # Check if session has enough frames
            if len(session_images) < min_frames_per_session:
                logging.warning(f"Session skipped due to insufficient frames: {session_path}")
                continue

            # Add session data to global lists
            images.extend(session_images)
            sensors.extend(session_sensors)
            labels.extend(session_labels)

    images = np.array(images, dtype=np.float32)
    sensors = np.array(sensors, dtype=np.float32)
    labels = utils.to_categorical(np.array(labels, dtype=np.int32), num_classes=2)

    logging.info(f"Loaded {len(labels)} samples (including augmented) from {data_dir}.")
    return images, sensors, labels


# In[187]:


img = preprocessing.image.load_img('extracted_frames/workspace/bad_pose/10/frame_0000.jpg', target_size=IMAGE_SIZE)
# print(img)
img_array = preprocessing.image.img_to_array(img)
# print(img_array)
augmented_imgs = [img_array] + [augment_image(img) for _ in range(10)]

# store teh augmented images in a directory
augmented_dir = "augmented_images"
os.makedirs(augmented_dir, exist_ok=True)
for i, img in enumerate(augmented_imgs):
    img_path = os.path.join(augmented_dir, f"augmented_{i}.jpg")
    img_pil = preprocessing.image.array_to_img(img)
    img_pil.save(img_path)


# In[190]:


def split_data(images, sensors, labels, validation_split=0.2, test_split=0.1):
    """
    Splits the dataset into training, testing, and validation sets.
    
    Args:
        images (np.ndarray): Array of image data.
        sensors (np.ndarray): Array of sensor data.
        labels (np.ndarray): Array of labels.
        validation_split (float): Proportion of the data to use for validation.
        test_split (float): Proportion of the data to use for testing.
        
    Returns:
        Tuple: (X_img_train, X_img_val, X_img_test, 
                X_sensor_train, X_sensor_val, X_sensor_test, 
                y_train, y_val, y_test)
    """
    dataset_size = len(images)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - test_size - val_size

    # Shuffle indices
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    # Split indices
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    # Slice data
    X_img_train = images[train_indices]
    X_img_val = images[val_indices]
    X_img_test = images[test_indices]
    X_sensor_train = sensors[train_indices]
    X_sensor_val = sensors[val_indices]
    X_sensor_test = sensors[test_indices]
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    return (X_img_train, X_img_val, X_img_test, 
            X_sensor_train, X_sensor_val, X_sensor_test, 
            y_train, y_val, y_test)


# In[191]:


images, sensors, labels = load_data(DATA_DIR)
# Split data using custom splitting logic

print(
    f"Loaded data: Images - {images.shape}, Sensors - {sensors.shape}, Labels - {labels.shape}"
)


# In[192]:


X_img_train, X_img_val, X_img_test, X_sensor_train, X_sensor_val, X_sensor_test, y_train, y_val, y_test = split_data(
    images, sensors, labels, validation_split=0.2, test_split=0.1
)

# Image input
image_input = layers.Input(shape=(224, 224, 3))

base_model = applications.MobileNetV2(
    weights="imagenet", include_top=False, input_tensor=image_input
)

image_features = layers.GlobalAveragePooling2D()(base_model.output)

# Sensor input
sensor_input = layers.Input(shape=(6,))  # 3 gyroscope + 3 accelerometer features
sensor_features = layers.Dense(
    64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
)(sensor_input)

# Combine features
combined = layers.concatenate([image_features, sensor_features])
combined = layers.Dropout(0.5)(combined)  # Add dropout before the dense layers
combined = layers.Dense(128, activation="relu")(combined)

output = layers.Dense(2, activation="softmax")(combined)

print(
    "Num GPUs Available: ", len(tf.config.list_physical_devices("GPU"))
)  # check if GPU is available
# check which gpu is running
print("The current used GPU is: ", tf.test.gpu_device_name())

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = optimizers.Adam(learning_rate=lr_schedule)


model = models.Model(inputs=[image_input, sensor_input], outputs=output)


model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:

# planning to add it but still not sure about the implementation or where to place it
# k fold cross validation
k = 5
num_val_samples = len(X_img_train) // k
num_epochs = 50


# In[205]:


class_weight = {0: 1.0, 1: 2.0}

history = model.fit(
    [X_img_train, X_sensor_train],
    y_train,
    validation_data=([X_img_val, X_sensor_val], y_val),
    class_weight=class_weight,
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)


# In[214]:


test_loss, test_accuracy = model.evaluate(
    [X_img_test, X_sensor_test],
    y_test,
    batch_size=32,
    verbose=1,
)
print(f"Validation Accuracy: {test_accuracy * 100:.2f}%")


# In[216]:


test_loss, test_accuracy


# In[213]:


model.save("multi_modal_posture_model.h5")
# save as tensorflow light model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("multi_modal_posture_model.tflite", "wb") as f:
    f.write(tflite_model)


# In[ ]:


impored_model = models.load_model("multi_modal_posture_model.h5")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "C:\\Users\\obada\\Desktop\\HoldWise Dataset\\posture_classifier.ipynb"')
get_ipython().system('jupyter nbconvert --to html "C:\\Users\\obada\\Desktop\\HoldWise Dataset\\posture_classifier.ipynb"')

