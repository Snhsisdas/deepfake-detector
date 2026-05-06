# DeepFake Detection System 🕵️

This repository contains an end-to-end DeepFake detection system that utilizes a hybrid Spatial-Temporal deep learning architecture to identify manipulated video media. The system extracts facial sequences and analyzes them for spatial artifacts and temporal inconsistencies using a combination of **MobileNetV2** (CNN) and **LSTM** (RNN). 

## 🚀 Features
- **Robust Face Extraction:** Uses MTCNN for high-quality facial cropping from video frames.
- **Spatial Analysis:** Leverages MobileNetV2 to extract features from individual frames efficiently.
- **Temporal Modeling:** Employs an LSTM network to detect unnatural transitions or flickering across frames.
- **Interactive UI:** A Streamlit web application for easy video uploading, processing, and visualization.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Snhsisdas/deepfake-detector.git
   cd deepfake-detector
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 How to Run

**1. Run the Web Interface (Streamlit App):**
To start the user interface and analyze a video, run:
```bash
streamlit run app.py
```
This will open the app in your default web browser where you can upload a `.mp4`, `.avi`, or `.mov` file to test it.

**2. Train the Model:**
If you want to train the model from scratch, ensure you have a dataset structured inside a `dataset/` folder with `real/` and `fake/` subdirectories. Then run:
```bash
python train.py
```

## 📁 Project Structure
- `app.py`: Streamlit application for the user interface.
- `model.py`: Defines the PyTorch `DeepFakeDetector` (CNN + LSTM).
- `face_extraction.py`: Contains the `FaceExtractor` class utilizing MTCNN.
- `train.py`: The training loop and validation pipeline.
- `dataset.py`: PyTorch `Dataset` class for processing video frames.
- `project_report.md`: Detailed report on the methodology and architecture.
