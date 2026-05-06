import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from face_extraction import FaceExtractor

class DeepFakeDataset(Dataset):
    def __init__(self, data_dir, num_frames=20, image_size=224, transform=None):
        """
        PyTorch Dataset for DeepFake Video Classification.
        :param data_dir: Directory containing 'real' and 'fake' subfolders.
        :param num_frames: Number of frames to extract per video.
        :param image_size: Size of the face crops.
        :param transform: PyTorch transforms to apply to each frame.
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.extractor = FaceExtractor(image_size=image_size)
        
        self.video_paths = []
        self.labels = [] # 0 for real, 1 for fake
        
        # Load real videos
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_paths.append(os.path.join(real_dir, file))
                    self.labels.append(0)
                    
        # Load fake videos
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_paths.append(os.path.join(fake_dir, file))
                    self.labels.append(1)
                    
        if transform is None:
            # Default transform for pre-trained ImageNet models
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract faces (returns list of PIL Images)
        face_images = self.extractor.extract_faces_from_video(video_path, num_frames=self.num_frames)
        
        # If no frames could be extracted (e.g., corrupt video), return a zero tensor
        if not face_images or len(face_images) < self.num_frames:
            # Fallback tensor
            frames_tensor = torch.zeros((self.num_frames, 3, self.image_size, self.image_size))
        else:
            # Apply transforms and stack
            frame_tensors = [self.transform(img) for img in face_images]
            frames_tensor = torch.stack(frame_tensors)
            
        return frames_tensor, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Test dataset
    if os.path.exists("dataset"):
        dataset = DeepFakeDataset("dataset", num_frames=5)
        print(f"Total videos found: {len(dataset)}")
        if len(dataset) > 0:
            frames, label = dataset[0]
            print(f"Frames tensor shape: {frames.shape}")
            print(f"Label: {label}")
