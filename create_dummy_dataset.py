import os
import cv2
import numpy as np

def create_dummy_video(filepath, num_frames=30, width=640, height=480):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 15.0, (width, height))
    
    for _ in range(num_frames):
        # Generate a random frame
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Add a face-like circle so MTCNN might catch something, or at least it has structure
        cv2.circle(frame, (width//2, height//2), 100, (200, 150, 100), -1)
        out.write(frame)
        
    out.release()
    print(f"Created dummy video: {filepath}")

if __name__ == "__main__":
    os.makedirs("dataset/real", exist_ok=True)
    os.makedirs("dataset/fake", exist_ok=True)
    
    print("Generating dummy dataset for testing pipeline...")
    create_dummy_video("dataset/real/real_video_1.mp4")
    create_dummy_video("dataset/fake/fake_video_1.mp4")
    
    print("\nDummy dataset generated in 'dataset' folder.")
    print("Note: These videos contain random noise. MTCNN may not detect a face.")
    print("Please replace these with real videos from FaceForensics++ for actual training.")
