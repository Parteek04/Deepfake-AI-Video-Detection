import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Flatten, TimeDistributed, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

def build_improved_model(input_shape=(10, 299, 299, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    model = Sequential([
        TimeDistributed(base_model, name="TimeDist_Xception"),
        TimeDistributed(Flatten(), name="TimeDist_Flatten"),
        Dropout(0.5, name="Dropout_1"),
        Bidirectional(LSTM(128), name="BiLSTM"),  
        Dropout(0.5, name="Dropout_2"),
        Dense(64, activation='relu', name="Dense_64"),
        Dense(2, activation='softmax', name="Output_Softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

#To test and visualize the model
if __name__ == "__main__":
    print(" Building model...")
    model = build_improved_model()
    model.summary()  # Show architecture in terminal

#Save architecture diagram 
    try:
        plot_model(model, to_file="model_architecture.png", show_shapes=True)
        print("Model diagram saved to model_architecture.png")
    except Exception as e:
        print("Plotting skipped (install pydot & graphviz if needed).")
