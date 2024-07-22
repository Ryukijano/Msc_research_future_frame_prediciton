import cv2
import os
from pathlib import Path

def process_jigsaws_suturing(video_dir, transcriptions_dir, output_dir, size=64):
    """Preprocesses JIGSAWS Suturing dataset based on TPG-VAE paper."""

    video_path = Path(video_dir)
    transcriptions_path = Path(transcriptions_dir)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    selected_gestures = ['G2', 'G3', 'G4', 'G6']

    video_files = [f for f in video_path.iterdir() if f.is_file() and f.suffix == '.avi']

    for video_file in video_files:
        video_name = video_file.stem
        user_id = video_name.split('_')[1]  # Extract User ID
        capture_name = video_name.split('_')[2]  # Extract capture name

        # Create user folder
        user_folder = output_path / user_id
        os.makedirs(user_folder, exist_ok=True)

        # Remove 'capture1' or 'capture2' from the video name to match transcription file
        transcription_name = '_'.join(video_name.split('_')[:2])
        transcription_file = transcriptions_path / f'{transcription_name}.txt'

        if not transcription_file.exists():
            print(f"Warning: Transcription file not found for {transcription_name}. Skipping.")
            continue

        gesture_segments = []
        with open(transcription_file, 'r') as f:
            for line in f:
                start_frame, end_frame, gesture = line.strip().split()
                start_frame, end_frame = int(start_frame), int(end_frame)
                if gesture in selected_gestures:
                    gesture_segments.append((start_frame, end_frame, gesture))

        cap = cv2.VideoCapture(str(video_file))

        for start_frame, end_frame, gesture in gesture_segments:
            frame_count = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while frame_count <= end_frame:
                ret, frame = cap.read()
                if ret:
                    if frame_count % 2 == 0:
                        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
                        frame_filename = user_folder / f'{capture_name}_{frame_count:04d}.png'
                        cv2.imwrite(str(frame_filename), frame)
                    frame_count += 1
                else:
                    break

        cap.release()
        print(f"Processed {video_name} and extracted frames for gestures: {selected_gestures}")


if __name__ == '__main__':
    suturing_video_dir = r"C:\\VPTR_jigsaws\\Suturing\\Suturing\\video"
    suturing_transcriptions_dir = r"C:\\VPTR_jigsaws\\Suturing\\Suturing\\transcriptions"
    output_frames_dir = 'Suturing/frames'
    process_jigsaws_suturing(suturing_video_dir, suturing_transcriptions_dir, output_frames_dir)