# Deepfake-AI-Video-Detection
This project aims to detect deepfake videos by using AI to analyze facial inconsistencies. By training models on real and fake content, the system classifies videos as authentic or manipulated, helping combat misinformation and ensure digital media integrity.
🔧 How to Run the Project
To ensure smooth training and deployment of the model, follow the steps below:

⚙️ Recommended Environment
Due to the computational requirements of deepfake detection, we recommend using Kaggle for training, as it provides free access to GPUs. The model is built using TensorFlow and saved in Keras (.h5) format.

🧪 Training the Model
Clone this repository and upload it to your Kaggle notebook.

Run the scripts in the following order:

Data_Preparation.py – Loads and prepares the dataset.

Data_Augmentation.py – Applies augmentation techniques to improve model performance.

Model_Architecture.py – Defines the deep learning architecture.

Model_Training.py – Trains the model and saves it in Keras format.

Once training is complete, download the trained model from Kaggle to use locally.

🚀 Running the Real-Time Detection
After downloading the trained model, you can use it locally with Visual Studio Code or any other IDE. The UI is built using Streamlit for interactivity.

To run the real-time detection app:
streamlit run Real_Time_Detection.py
Ensure the following scripts have been executed in sequence before launching the detection script:

Data_Preparation.py

Data_Augmentation.py

Model_Architecture.py

Model_Training.py

Real_Time_Detection.py – for final real-time deepfake detection output.

📦 Download Pretrained Model (Optional)
If you want to skip training, you can directly download the pretrained model (trained by me) using the link below:

👉 https://www.kaggle.com/models/parteeek/deepfake-ai-video-detection-2025
