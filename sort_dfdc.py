import os
import json
import shutil

def sort_dfdc_videos(metadata_path, videos_dir, output_dir="dataset"):
    """
    Reads the DFDC metadata.json file and copies the videos into the
    dataset/real and dataset/fake folders so the PyTorch DataLoader can read them.
    """
    # Create output directories
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Could not read metadata.json: {e}")
        return

    # Move videos based on label
    print("Sorting videos into 'real' and 'fake' folders...")
    moved_count = 0
    for video_name, info in metadata.items():
        label = info.get("label")
        source_path = os.path.join(videos_dir, video_name)
        
        if not os.path.exists(source_path):
            continue
            
        if label == "REAL":
            dest_path = os.path.join(real_dir, video_name)
        elif label == "FAKE":
            dest_path = os.path.join(fake_dir, video_name)
        else:
            continue
            
        # Copy the file to the new destination
        shutil.copy(source_path, dest_path)
        moved_count += 1
        
        if moved_count % 50 == 0:
            print(f"Sorted {moved_count} videos...")
            
    print(f"\nDone! Successfully sorted {moved_count} videos.")
    print("You can now run 'python train.py' to train your model!")

if __name__ == "__main__":
    print("--- Kaggle DFDC Dataset Sorter ---")
    videos_dir = input("Enter the full path to the extracted 'train_sample_videos' folder: ").strip()
    
    # Remove quotes if the user dragged and dropped the folder into the terminal
    videos_dir = videos_dir.strip('"').strip("'")
    
    metadata_path = os.path.join(videos_dir, "metadata.json")
    
    if os.path.exists(metadata_path):
        sort_dfdc_videos(metadata_path, videos_dir)
    else:
        print(f"\nError: Could not find 'metadata.json' inside {videos_dir}")
        print("Make sure you are pointing to the folder that contains the .mp4 files AND the metadata.json file.")
