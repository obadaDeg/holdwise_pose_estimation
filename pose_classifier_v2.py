#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
# from tensorflow import keras
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[6]:


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


# In[7]:


DATASET_PATH = os.path.abspath("organized_data")  # Ensure absolute path
print("Dataset Path:", DATASET_PATH)

train_path = os.path.join(DATASET_PATH, "train")
test_path = os.path.join(DATASET_PATH, "test")


# In[8]:


def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
        return np.array(keypoints)
    else:
        return None  # If no pose is detected


# In[9]:


def load_dataset(dataset_path):
    data, labels = [], []
    class_map = {"good": 0, "bad": 1, "cant_determine": 2}
    
    for class_name, class_label in class_map.items():
        class_dir = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist. Skipping.")
            continue
        
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            keypoints = extract_keypoints(file_path)
            if keypoints is not None:
                data.append(keypoints)
                labels.append(class_label)
    
    return np.array(data), np.array(labels)


# In[10]:


X_train, y_train = load_dataset(train_path)
X_test, y_test = load_dataset(test_path)

valiadation_split = 0.2

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valiadation_split, random_state=42)


# In[11]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[12]:


X_train = np.nan_to_num(X_train)  # Replace NaN with 0
X_test = np.nan_to_num(X_test)


# In[13]:


X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)


# In[14]:


model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 classes
])


# In[15]:


optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[16]:


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)


# In[20]:


# model.evaluate(X_test, y_test)
loss, accuracy = model.evaluate(X_test, y_test)


# In[21]:


loss, accuracy


# In[18]:


model.save("pose_classification_model.h5")
converter = tf.lite.TFLiteConverter(model)
tflite_model = converter.convert()
with open("pose_classification_model.tflite", "wb") as f:
    f.write(tjson.dumps(tflite_model))


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "C:\\Users\\obada\\Desktop\\HoldWise Dataset\\pose_classifier_v2.ipynb"')
get_ipython().system('jupyter nbconvert --to html "C:\\Users\\obada\\Desktop\\HoldWise Dataset\\pose_classifier_v2.ipynb"')

