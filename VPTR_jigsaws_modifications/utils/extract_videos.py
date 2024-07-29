import cv2
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def extract_frames(video_file, output_path, size=64):
    """Extracts frames from a video and saves them to the output directory."""
    video_name = video_file.stem
    user_id = video_name.split('_')[1]
    capture_name = video_name.split('_')[2]

    user_folder = output_path / user_id / capture_name
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 2 == 0:  # Select every other frame
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            frame_filename = user_folder / f'{frame_count:04d}.png'
            cv2.imwrite(str(frame_filename), frame)
        frame_count += 1
    cap.release()

if __name__ == '__main__':
    suturing_video_dir = r"D:\VPTR_jigsaws\Suturing\Suturing\video"
    output_frames_dir = r"D:\VPTR_jigsaws\Suturing\frames"
    
    video_path = Path(suturing_video_dir)
    output_path = Path(output_frames_dir)
    os.makedirs(output_path, exist_ok=True)

    video_files = [f for f in video_path.iterdir() if f.is_file() and f.suffix == '.avi']

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(extract_frames, video_files, [output_path] * len(video_files)), total=len(video_files)))