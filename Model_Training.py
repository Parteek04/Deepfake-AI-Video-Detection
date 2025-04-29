import numpy as np
import os
from Model_Architecture import build_improved_model
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Compute class weights
y_train_labels = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))

# Build model (input size now 299x299 for MobileNetV2)
input_shape = (10, 299, 299, 3)
model = build_improved_model(input_shape=input_shape)

# Callbacks
checkpoint = ModelCheckpoint("deepfake_detection_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

# Train model with smaller batch size
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=2,  
    callbacks=[checkpoint, lr_scheduler],
    class_weight=class_weight_dict
)

# Save final model
save_dir = "/Users/poornimavaidya/Desktop/Masters/Spring2025/Adv_AI/ProjectCode/Model"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, "deepfake_detection_model_final.keras"))

print("✅ Model training completed and saved.")