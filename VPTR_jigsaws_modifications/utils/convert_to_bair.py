import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def move_frames_to_bair(gesture_folder, user_id, split, input_path, output_path, example_count):
    """Moves frames for a specific user and gesture to a BAIR-style example folder."""
    user_frames = sorted((input_path / split / gesture_folder.name).glob(f'{user_id}_*.png'))
    if user_frames:
        example_name = f'example_{example_count}'
        dest_folder = output_path / split / example_name
        os.makedirs(dest_folder, exist_ok=True)
        for i, frame_file in enumerate(user_frames):
            shutil.copy(str(frame_file), str(dest_folder / f'{i:04d}.png'))
        print(f"Moved frames for user {user_id}, gesture {gesture_folder.name} to {dest_folder}")

def convert_to_bair(input_dir, output_dir):
    """Converts TPG-VAE formatted frames to BAIR format."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for split in ['train', 'test']:
        split_path = output_path / split
        os.makedirs(split_path, exist_ok=True)
        example_count = 0
        tasks = []
        for gesture_folder in (input_path / split).iterdir():
            for user_id in ['B001', 'B002', 'B003', 'B004', 'B005', 'C001', 'C002', 'C003', 'C004', 'C005',
                        'D001', 'D002', 'D003', 'D004', 'D005', 'E001', 'E002', 'E003', 'E004', 'E005',
                        'F001', 'F002', 'F003', 'F004', 'F005', 'G001', 'G002', 'G003', 'G004', 'G005',
                        'H001', 'H003', 'H004', 'H005', 'I001', 'I002', 'I003', 'I004', 'I005']:
                tasks.append((gesture_folder, user_id, split, input_path, output_path, example_count))
                example_count += 1
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(move_frames_to_bair, *zip(*tasks)), total=len(tasks)))

if __name__ == '__main__':
    tpgvae_frames_dir = r"D:\VPTR_jigsaws\Suturing\frames_tpgvae"  # Output from the first script
    bair_format_dir = r"D:\VPTR_jigsaws\Suturing\bair_format_dir"  # Output directory for BAIR format
    convert_to_bair(tpgvae_frames_dir, bair_format_dir)