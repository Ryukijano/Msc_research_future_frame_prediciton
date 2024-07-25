import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_gesture_segment(user_capture_path, transcription_file, gesture, start_frame, end_frame, split, output_path):
    """Processes a single gesture segment and copies frames to the output directory."""
    for frame_num in range(start_frame, end_frame + 1, 2):  # Every other frame
        src_path = user_capture_path / f'{frame_num:04d}.png'
        if src_path.exists():
            dest_path = output_path / split / gesture / f'{user_capture_path.parent.name}_{frame_num:04d}.png'
            os.makedirs(output_path / split / gesture, exist_ok=True)
            shutil.copy(str(src_path), str(dest_path))

def create_tpgvae_dataset(frames_dir, transcriptions_dir, output_dir):
    """Organizes extracted frames into TPG-VAE format."""

    frames_path = Path(frames_dir)
    transcriptions_path = Path(transcriptions_dir)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    selected_gestures = ['G2', 'G3', 'G4', 'G6']
    train_users = ['B001', 'B002', 'C001', 'C002', 'D001', 'D002']
    test_users = ['E001', 'E002', 'E003', 'E004', 'E005', 'F001', 'F002', 'F003', 'F004', 'F005',
                  'G001', 'G002', 'G003', 'G004', 'G005', 'H001', 'H003', 'H004', 'H005',
                  'I001', 'I002', 'I003', 'I004', 'I005']

    tasks = []
    for user_id in train_users + test_users:
        for capture in ['capture1', 'capture2']:
            user_capture_path = frames_path / user_id / capture
            if not user_capture_path.exists():
                continue

            transcription_file = transcriptions_path / f'Suturing_{user_id}.txt'
            if not transcription_file.exists():
                print(f"Warning: Transcription file not found for Suturing_{user_id}. Skipping.")
                continue

            split = 'train' if user_id in train_users else 'test'

            with open(transcription_file, 'r') as f:
                for line in f:
                    start_frame, end_frame, gesture = line.strip().split()
                    start_frame, end_frame = int(start_frame), int(end_frame)
                    if gesture in selected_gestures:
                        tasks.append((user_capture_path, transcription_file, gesture, start_frame, end_frame, split, output_path))

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_gesture_segment, *zip(*tasks)), total=len(tasks)))

if __name__ == '__main__':
    extracted_frames_dir = r"D:\VPTR_jigsaws\Suturing\frames"  # Output from the first script
    suturing_transcriptions_dir = r"D:\VPTR_jigsaws\Suturing\Suturing\transcriptions"
    tpgvae_format_dir = r"D:\VPTR_jigsaws\Suturing\frames_tpgvae"  # Output directory for TPG-VAE format
    create_tpgvae_dataset(extracted_frames_dir, suturing_transcriptions_dir, tpgvae_format_dir)