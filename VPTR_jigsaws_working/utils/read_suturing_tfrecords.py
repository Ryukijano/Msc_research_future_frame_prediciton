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
    """

    suturing_path = Path(suturing_dir)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    video_files = [f for f in suturing_path.iterdir() if f.is_file() and f.suffix == '.avi']

    for video_idx, video_file in enumerate(video_files):
        video_name = video_file.stem  # Get filename without extension
        example_folder = output_path / f'example_{video_idx}'  # BAIR-like naming
        os.makedirs(example_folder, exist_ok=True)

        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame_filename = example_folder / f'{frame_count:04d}.png'
                cv2.imwrite(str(frame_filename), frame)
                frame_count += 1
            else:
                break

        cap.release()
        print(f"Converted {video_name} to frames in {example_folder}")

def resize_images(data_dir, size=64):
    """Resizes images within the BAIR-like dataset to the specified size.

    Args:
        data_dir (str): Path to the directory containing the BAIR-like dataset.
        size (int): Desired image size (square).
    """

    data_path = Path(data_dir)
    for example_folder in tqdm(data_path.glob('example_*'), desc='Resizing...'):
        for img_file in example_folder.glob('*.png'):
            img = tf.io.read_file(str(img_file))
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [size, size], method=tf.image.ResizeMethod.BICUBIC)
            img = tf.image.convert_image_dtype(img, tf.uint8)
            tf.io.write_file(str(img_file), tf.image.encode_png(img))

if __name__ == '__main__':
    suturing_dataset_dir = r"D:\VPTR_jigsaws\Suturing\Suturing\video"  
    output_directory = "Suturing/frames" 
    read_suturing_videos(suturing_dataset_dir, output_directory)

    # Resize images to 64x64
    resize_images(output_directory, size=128)