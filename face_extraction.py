import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

class FaceExtractor:
    def __init__(self, device='cpu', image_size=224, margin=20):
        """
        Initializes the MTCNN face extractor.
        :param device: 'cuda' or 'cpu'
        :param image_size: The target size of the output face crops (default 224 for ResNet/MobileNet)
        :param margin: Margin to add around the detected face bounding box
        """
        self.device = device
        self.image_size = image_size
        # keep_all=False ensures we only take the most prominent face
        self.mtcnn = MTCNN(keep_all=False, device=self.device, image_size=self.image_size, margin=margin)

    def extract_faces_from_video(self, video_path, num_frames=20):
        """
        Reads a video, samples num_frames evenly, and extracts faces.
        If a face is not found, it takes a center crop as a fallback.
        :param video_path: Path to the video file
        :param num_frames: Number of frames to extract from the video
        :return: List of PIL Images containing cropped faces (or center crops)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return []

        # Calculate indices of frames to extract
        frame_idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        extracted_faces = []
        
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Detect face and crop
            face_tensor = self.mtcnn(pil_img)
            
            if face_tensor is not None:
                # MTCNN returns a normalized tensor, but we might want to return PIL images
                # for subsequent generic transforms (like data augmentation) in the dataset.
                # Let's extract the bounding box manually so we can return a PIL image.
                boxes, probs = self.mtcnn.detect(pil_img)
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    # Apply margin
                    margin = self.mtcnn.margin
                    x1 = max(0, box[0] - margin/2)
                    y1 = max(0, box[1] - margin/2)
                    x2 = min(pil_img.width, box[2] + margin/2)
                    y2 = min(pil_img.height, box[3] + margin/2)
                    
                    face_img = pil_img.crop((x1, y1, x2, y2))
                    face_img = face_img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                    extracted_faces.append(face_img)
                else:
                    extracted_faces.append(self._get_center_crop(pil_img))
            else:
                # Fallback if no face detected
                extracted_faces.append(self._get_center_crop(pil_img))
                
        cap.release()
        
        # Pad if we didn't get enough frames
        while len(extracted_faces) > 0 and len(extracted_faces) < num_frames:
            extracted_faces.append(extracted_faces[-1]) # Duplicate last frame
            
        return extracted_faces
        
    def _get_center_crop(self, pil_img):
        """Fallback method to get a center crop if face is not found."""
        w, h = pil_img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        right = (w + min_dim) / 2
        bottom = (h + min_dim) / 2
        crop = pil_img.crop((left, top, right, bottom))
        return crop.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

if __name__ == "__main__":
    # Test the extractor
    extractor = FaceExtractor()
    # Assuming dummy dataset is created
    import os
    if os.path.exists("dataset/real/real_video_1.mp4"):
        faces = extractor.extract_faces_from_video("dataset/real/real_video_1.mp4", num_frames=5)
        print(f"Extracted {len(faces)} faces from test video.")
        if len(faces) > 0:
            print(f"Face image size: {faces[0].size}")
