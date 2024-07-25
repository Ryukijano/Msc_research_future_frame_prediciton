import cv2
import os
from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def read_suturing_videos(suturing_dir, output_dir, size=64):
    """Reads Suturing videos, extracts frames, and saves them in BAIR-like format.

    Args:
        suturing_dir (str): Path to the directory containing Suturing .avi files.
        output_dir (str): Path to the output directory.
        size (int): Desired output size for the images (square).
    """

    suturing_path = Path(suturing_dir)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    video_files = [f for f in suturing_path.iterdir() if f.is_file() and f.suffix == '.avi']

    for video_idx, video_file in enumerate(video_files):
        video_name = video_file.stem
        example_folder = output_path / f'example_{video_idx}'
        os.makedirs(example_folder, exist_ok=True)

        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # Resize the frame using bilinear interpolation
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR) 
                frame_filename = example_folder / f'{frame_count:04d}.png'
                cv2.imwrite(str(frame_filename), frame)
                frame_count += 1
            else:
                break

        cap.release()
        print(f"Converted {video_name} to frames in {example_folder}")

if __name__ == '__main__':
    suturing_dataset_dir = r"D:\VPTR_jigsaws\Suturing\Suturing\video"
    output_directory = "D:\\VPTR_jigsaws\\Suturing\\frames" 
    read_suturing_videos(suturing_dataset_dir, output_directory, size=64)  # Set output size to 128x128