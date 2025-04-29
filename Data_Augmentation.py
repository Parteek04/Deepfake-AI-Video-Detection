import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Load preprocessed training data
X_train = np.load("X_train.npy")  
y_train = np.load("y_train.npy")  

print("Original training data:", X_train.shape)

# Define augmentation generator
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

# Function to augment a sequence of frames
def augment_video(frames):
    return np.array([datagen.random_transform(frame) for frame in frames])

# How many augmented copies? (1 = double dataset)
AUGMENT_FACTOR = 1

augmented_data = []
augmented_labels = []

print("Augmenting data...")
for _ in range(AUGMENT_FACTOR):
    for i in tqdm(range(len(X_train))):
        augmented_frames = augment_video(X_train[i])
        augmented_data.append(augmented_frames)
        augmented_labels.append(y_train[i])

# Convert to numpy arrays
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)

# Combine with original
X_train_augmented = np.concatenate((X_train, augmented_data), axis=0)
y_train_augmented = np.concatenate((y_train, augmented_labels), axis=0)

# Save augmented data
np.save("X_train_augmented.npy", X_train_augmented)
np.save("y_train_augmented.npy", y_train_augmented)

print("✅ Augmentation complete.")
print(f"Final training shape: {X_train_augmented.shape}, {y_train_augmented.shape}")
