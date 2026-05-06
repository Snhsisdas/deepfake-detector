import os
import torch
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import tempfile
import cv2

from model import DeepFakeDetector
from face_extraction import FaceExtractor

# UI Configuration
st.set_page_config(page_title="DeepFake Detector", page_icon="🕵️", layout="wide")

# Constants
NUM_FRAMES = 10
IMAGE_SIZE = 224
MODEL_PATH = "model_checkpoint.pth"

@st.cache_resource
def load_model():
    model = DeepFakeDetector()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        st.sidebar.success("Model loaded successfully.")
    else:
        st.sidebar.warning("Model checkpoint not found. Using untrained weights for demonstration.")
        
    model.eval()
    return model, device

@st.cache_resource
def load_extractor():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return FaceExtractor(device=device, image_size=IMAGE_SIZE)

def predict(video_path, model, extractor, device):
    # Extract faces
    faces = extractor.extract_faces_from_video(video_path, num_frames=NUM_FRAMES)
    
    if not faces or len(faces) == 0:
        return None, None, []
        
    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_tensors = [transform(img) for img in faces]
    # Handle case where fewer frames were extracted
    while len(frame_tensors) < NUM_FRAMES:
        frame_tensors.append(frame_tensors[-1])
        faces.append(faces[-1])
        
    input_tensor = torch.stack(frame_tensors).unsqueeze(0).to(device) # Shape: (1, seq_len, C, H, W)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze()
        fake_prob = probabilities[1].item()
        real_prob = probabilities[0].item()
        
    prediction = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(fake_prob, real_prob) * 100
    
    return prediction, confidence, faces

# Main UI
st.title("🕵️ DeepFake Detection via Spatial & Temporal Analysis")
st.markdown("Upload a video to analyze it for DeepFake artifacts. This tool extracts sequences of faces and analyzes them using a CNN (MobileNetV2) and LSTM.")

model, device = load_model()
extractor = load_extractor()

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close() # Close the file handle so Windows doesn't lock it
    video_path = tfile.name
    
    # Display the video directly from the uploaded file buffer to prevent locking
    st.video(uploaded_file)
    
    if st.button("Analyze Video"):
        with st.spinner('Analyzing video (Extracting faces & processing)...'):
            prediction, confidence, faces = predict(video_path, model, extractor, device)
            
            if prediction is None:
                st.error("Could not extract any frames from the video.")
            else:
                st.markdown("---")
                st.header("Analysis Results")
                
                # Display Results
                if prediction == "FAKE":
                    st.error(f"Prediction: **{prediction}** ({confidence:.2f}% confidence)")
                else:
                    st.success(f"Prediction: **{prediction}** ({confidence:.2f}% confidence)")
                    
                st.markdown("### Extracted Sequence")
                st.markdown("The following faces were extracted and analyzed sequentially to detect temporal anomalies:")
                
                # Show extracted faces
                cols = st.columns(len(faces))
                for i, col in enumerate(cols):
                    with col:
                        st.image(faces[i], use_container_width=True, caption=f"Frame {i+1}")
                        
    # Clean up temp file
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except PermissionError:
            pass # Windows might still be holding a lock, OS will clean it up later
