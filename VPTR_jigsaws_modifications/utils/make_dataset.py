
import os
import shutil
from pathlib import Path
import random

def split_suturing_dataset(frames_dir, meta_file, output_dir):
    """Splits the Suturing dataset into train/val/test sets based on user ID (LOUO).

    Args:
        frames_dir (str): Path to the directory containing the extracted frames in BAIR-like format.
        meta_file (str): Path to the 'meta_file_Suturing.txt' file.
        output_dir (str): Path to the output directory for the splits.
    """

    frames_path = Path(frames_dir)
    output_path = Path(output_dir)

    # Create train/val/test directories
    os.makedirs(output_path / 'train', exist_ok=True)
    os.makedirs(output_path / 'val', exist_ok=True)
    os.makedirs(output_path / 'test', exist_ok=True)

    # Read user IDs from the meta file
    user_ids = []
    with open(meta_file, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Check if the line is not empty
                parts = line.split('_')
                if len(parts) >= 2:  # Check if there are at least two parts
                    user_id = parts[1] # Extract user ID (e.g., 'B001')
                    user_ids.append(user_id)
                else:
                    print(f"Warning: Invalid line format: {line}")
            else:
                print("Warning: Skipping empty line.")

    # Perform LOUO split
    for test_user_id in user_ids:
        print(f"Creating split for test user: {test_user_id}")
        for example_folder in frames_path.glob('example_*'):
            video_name = example_folder.name  # e.g., 'example_0'
            user_id = '_'.join(video_name.split('_')[1:3])  # e.g., 'B001_capture1'
            user_id = user_id.split('_')[0] # e.g., 'B001'
            if user_id == test_user_id:
                shutil.move(str(example_folder), str(output_path / 'test' / video_name))
            else:
                # 80% for training, 20% for validation
                if random.random() < 0.8:
                    shutil.move(str(example_folder), str(output_path / 'train' / video_name))
                else:
                    shutil.move(str(example_folder), str(output_path / 'val' / video_name))

if __name__ == '__main__':
    frames_directory = 'C:\\VPTR_jigsaws\\jigsaws_suturing\\frames'  # Your frames directory
    meta_file_path = 'C:\\VPTR_jigsaws\\Suturing\\Suturing\\meta_file_Suturing.txt'  # Path to the meta file
    output_split_dir = 'C:\\VPTR_jigsaws\\jigsaws_suturing\\frames_split'  # Output directory for splits
    split_suturing_dataset(frames_directory, meta_file_path, output_split_dir)