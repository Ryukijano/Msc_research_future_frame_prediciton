import cv2
import os
from pathlib import Path

def process_jigsaws_tpgvae(videos_dir, transcriptions_dir, output_dir, size=64):
    """Preprocesses JIGSAWS Suturing videos into the TPG-VAE format."""

    video_path = Path(videos_dir)
    transcriptions_path = Path(transcriptions_dir)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    selected_gestures = ['G2', 'G3', 'G4', 'G6']
    train_users = ['B001', 'B002', 'C001', 'C002', 'D001', 'D002']
    test_users = ['E001', 'E002', 'E003', 'E004', 'E005', 'F001', 'F002', 'F003', 'F004', 'F005', 
                  'G001', 'G002', 'G003', 'G004', 'G005', 'H001', 'H003', 'H004', 'H005', 
                  'I001', 'I002', 'I003', 'I004', 'I005']

    video_files = [f for f in video_path.iterdir() if f.is_file() and f.suffix == '.avi']

    for video_file in video_files:
        video_name = video_file.stem
        user_id = video_name.split('_')[1]
        transcription_file = transcriptions_path / f'{video_name.split("_")[0]}_{user_id}.txt'

        if not transcription_file.exists():
            print(f"Warning: Transcription file not found for {video_name}. Skipping.")
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
                        # Determine train/test folder based on user ID
                        if user_id in train_users:
                            split = 'train'
                        elif user_id in test_users:
                            split = 'test'
                        else:
                            print(f"Warning: User ID {user_id} not found in train or test sets. Skipping.")
                            continue

                        frame_filename = output_path / split / gesture / f'{user_id}_{frame_count:04d}.png'
                        os.makedirs(output_path / split / gesture, exist_ok=True)
                        cv2.imwrite(str(frame_filename), frame)
                    frame_count += 1
                else:
                    break

        cap.release()
        print(f"Processed {video_name} and extracted frames for gestures: {selected_gestures}")

if __name__ == '__main__':
    suturing_video_dir = r"D:\VPTR_jigsaws\Suturing\Suturing\video"
    suturing_transcriptions_dir = r"D:\VPTR_jigsaws\Suturing\Suturing\transcriptions"
    output_frames_dir = 'Suturing/frames_tpgvae'  # Output directory for TPG-VAE format
    process_jigsaws_tpgvae(suturing_video_dir, suturing_transcriptions_dir, output_frames_dir)