# DeepFake Detection System: Project Report

## 1. Introduction
The rapid advancement of generative artificial intelligence has led to a surge in highly realistic manipulated media, commonly known as DeepFakes. These forged videos pose significant risks to personal identity, media integrity, and information security. The objective of this project is to develop an end-to-end DeepFake detection system capable of identifying manipulated video media. To achieve this, the project implements a hybrid deep learning architecture that analyzes both spatial artifacts within individual frames and temporal inconsistencies across a sequence of frames. The system is designed to be efficient enough to run locally on consumer-grade hardware (such as a 4GB VRAM GPU) while providing a user-friendly interface for video analysis.

## 2. Methods
The DeepFake detection pipeline is composed of three primary stages: face extraction, spatial-temporal feature analysis, and a user interface for deployment.

**2.1. Data Preprocessing and Face Extraction**
To focus the model's attention on the regions most likely to contain manipulation artifacts, facial extraction is performed on the input videos. 
- The system uniformly samples a fixed number of frames (e.g., 10 or 20 frames) across the duration of the video using OpenCV.
- Face detection and cropping are handled by **MTCNN** (Multi-task Cascaded Convolutional Networks). A margin is added around the detected bounding boxes to capture contextual information.
- In cases where a face cannot be detected in a frame, the system robustly falls back to a center-crop strategy. All extracted faces are resized to 224x224 pixels and normalized before being fed into the neural network.

**2.2. Spatial and Temporal Modeling**
The core neural network architecture (`DeepFakeDetector`) is a hybrid model combining Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN):
- **Spatial Feature Extraction:** A pre-trained **MobileNetV2** is utilized as the CNN backbone. The final classification layer is removed, allowing the network to act as a feature extractor. MobileNetV2 was selected for its lightweight architecture, ensuring low memory consumption and fast inference times. It converts each 224x224 face crop into a dense 1280-dimensional feature vector.
- **Temporal Modeling:** To capture unnatural transitions, blinking anomalies, or temporal flickering common in DeepFakes, the sequence of spatial feature vectors is fed into an **LSTM** (Long Short-Term Memory) network. 
- **Classification Head:** The final hidden state of the LSTM is passed through fully connected layers (with Dropout and ReLU activation) to produce the final classification probabilities for two classes: "Real" and "Fake".

**2.3. Training Pipeline and Deployment**
- The model is trained using PyTorch with the Adam optimizer and Cross-Entropy Loss. The training script includes a validation loop to track accuracy and automatically saves the best-performing model checkpoint (`model_checkpoint.pth`).
- The system is deployed via a **Streamlit** web application. The interactive UI allows users to upload videos, processes the file through the extraction and inference pipeline, and visualizes the sequence of extracted faces alongside the model's final prediction and confidence score.

## 3. Results
The implemented system successfully processes raw video files end-to-end. During inference, the web application accurately displays the extracted face sequence, providing transparency into the model's focus area. The integration of MTCNN ensures high-quality face crops, which directly contributes to the quality of the features extracted by MobileNetV2. When evaluating uploaded videos, the Streamlit app seamlessly handles temporary file management, runs inference via the trained model checkpoint, and outputs a clear "REAL" or "FAKE" classification with a percentage-based confidence score. 

## 4. Discussion
The hybrid CNN-LSTM approach addresses the limitations of relying solely on frame-by-frame analysis. While spatial models can detect pixel-level warping or blending errors, the addition of the LSTM allows the system to identify subtle temporal inconsistencies that temporal-agnostic models miss. 

The choice of MobileNetV2 is a critical design decision that balances performance with computational efficiency, enabling the pipeline to execute smoothly on local GPUs with limited VRAM. Furthermore, the robust fallback mechanism in the face extraction module ensures that the pipeline does not crash when dealing with corrupt frames, heavy occlusion, or extreme head poses, maintaining the fixed sequence length required by the LSTM.

## 5. Conclusion
This project successfully delivers a functional, end-to-end DeepFake detection tool. By combining MTCNN for precise face localization, MobileNetV2 for efficient spatial feature extraction, and LSTM for temporal sequence modeling, the system effectively captures the multifaceted artifacts present in manipulated videos. The Streamlit-based user interface wraps the complex deep learning pipeline into an accessible application, making it easy to test and demonstrate the model's capabilities. Future enhancements could include integrating Vision Transformers, expanding the training dataset, or implementing real-time webcam analysis.
