import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

# Dataset paths
REAL_PATH = "/Users/poornimavaidya/Desktop/Masters/Spring2025/Adv_AI/ProjectCode/Data_set_100/archive/Real"
FAKE_PATH = "/Users/poornimavaidya/Desktop/Masters/Spring2025/Adv_AI/ProjectCode/Data_set_100/archive/Fake"
OUTPUT_FRAME_SIZE = (299, 299)  
FRAME_COUNT = 10
MAX_VIDEOS = 150

# Helper to check video file extensions
def is_video_file(filename):
    return filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

# Extract frames from video
def extract_frames(video_path, output_size=(299, 299), frame_count=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Failed to open video {video_path}")
        return np.array([])

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frame_count, 1)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        frames.append(frame)

    cap.release()
    return np.array(frames)

data = []
labels = []
failed_videos = []

print("Processing real videos...")
real_videos = [f for f in os.listdir(REAL_PATH) if is_video_file(f)][:MAX_VIDEOS]
for video_file in tqdm(real_videos):
    full_path = os.path.join(REAL_PATH, video_file)
    frames = extract_frames(full_path)
    if len(frames) != FRAME_COUNT:
        print(f"Skipping incomplete or failed video: {video_file}")
        failed_videos.append(full_path)
        continue
    data.append(frames)
    labels.append(0)  # REAL

print("Processing fake videos...")
fake_videos = [f for f in os.listdir(FAKE_PATH) if is_video_file(f)][:MAX_VIDEOS]
for video_file in tqdm(fake_videos):
    full_path = os.path.join(FAKE_PATH, video_file)
    frames = extract_frames(full_path)
    if len(frames) != FRAME_COUNT:
        print(f"Skipping incomplete or failed video: {video_file}")
        failed_videos.append(full_path)
        continue
    data.append(frames)
    labels.append(1)  # FAKE

# Save failed videos log
with open("failed_videos.txt", "w") as f:
    for file_path in failed_videos:
        f.write(file_path + "\n")

# Convert to numpy arrays and shuffle
data = np.array(data)
labels = np.array(labels)
data, labels = shuffle(data, labels, random_state=42)

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Normalize
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Save datasets
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Data prep complete.")
print(f"Failed videos logged in failed_videos.txt ({len(failed_videos)} files)")
