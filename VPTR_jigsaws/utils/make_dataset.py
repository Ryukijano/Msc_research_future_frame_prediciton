import os
import cv2
import random
from pathlib import Path
from PIL import Image

def preprocess_and_split_dataset(videos_dir, meta_file, transcriptions_dir, output_dir, frame_size=(64, 64), sequence_length=30):
    print("Starting dataset preprocessing and splitting...")
    videos_path = Path(videos_dir)
    output_path = Path(output_dir)
    transcriptions_path = Path(transcriptions_dir)

    os.makedirs(output_path / 'train', exist_ok=True)
    os.makedirs(output_path / 'test', exist_ok=True)

    user_gestures = []
    print("Reading meta file...")
    with open(meta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) >= 2:
                    video_name, skill_level = parts[0], parts[1]
                    user_id = video_name.split('_')[1]
                    user_gestures.append((video_name, user_id, skill_level))
                else:
                    print(f"Warning: Invalid line format: {line}")

    unique_users = sorted(set(user_id for _, user_id, _ in user_gestures))
    train_users = unique_users[:6]
    test_users = unique_users[6:]
    print(f"Train users: {train_users}")
    print(f"Test users: {test_users}")

    test_sequences = {gesture: [] for gesture in ['2', '3', '4', '6']}
    total_videos = len(user_gestures)
    for idx, (video_name, user_id, _) in enumerate(user_gestures, 1):
        print(f"Processing video {idx}/{total_videos}: {video_name}")
        video_path_1 = videos_path / f"{video_name}_capture1.avi"
        video_path_2 = videos_path / f"{video_name}_capture2.avi"
        transcription_path = transcriptions_path / f"{video_name}.txt"
        
        if video_path_1.exists() and video_path_2.exists() and transcription_path.exists():
            frames_1 = extract_and_preprocess_frames(video_path_1, frame_size)
            frames_2 = extract_and_preprocess_frames(video_path_2, frame_size)
            frames = [Image.blend(f1, f2, 0.5) for f1, f2 in zip(frames_1, frames_2)]  # Blend the two views
            
            gesture_segments = read_transcription(transcription_path)
            
            for start_frame, end_frame, gesture in gesture_segments:
                if gesture in ['G2', 'G3', 'G4', 'G6']:
                    gesture = gesture[1]  # Remove 'G' prefix
                    segment_frames = frames[start_frame:end_frame+1]
                    sequences = create_sequences(segment_frames, sequence_length)
                    
                    for seq_idx, sequence in enumerate(sequences):
                        if user_id in train_users:
                            save_sequence(sequence, output_path / 'train' / f"{video_name}_G{gesture}_seq{seq_idx}")
                        else:
                            test_sequences[gesture].append((sequence, f"{video_name}_G{gesture}_seq{seq_idx}"))
        else:
            print(f"Warning: Video files or transcription file not found for {video_name}")

    print("Balancing test sequences...")
    min_gesture_count = min(len(seqs) for seqs in test_sequences.values())
    for gesture, sequences in test_sequences.items():
        selected_sequences = random.sample(sequences, min_gesture_count)
        for sequence, seq_name in selected_sequences:
            save_sequence(sequence, output_path / 'test' / seq_name)

    print("Dataset preprocessing and splitting completed.")

def extract_and_preprocess_frames(video_path, frame_size):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def read_transcription(transcription_path):
    gesture_segments = []
    with open(transcription_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start_frame, end_frame, gesture = int(parts[0]), int(parts[1]), parts[2]
                gesture_segments.append((start_frame, end_frame, gesture))
    return gesture_segments

def create_sequences(frames, sequence_length):
    return [frames[i:i+sequence_length] for i in range(0, len(frames) - sequence_length + 1, sequence_length // 2)]

def save_sequence(sequence, output_path):
    os.makedirs(output_path, exist_ok=True)
    for i, frame in enumerate(sequence):
        frame.save(output_path / f"frame_{i:03d}.png")

if __name__ == '__main__':
    videos_directory = r'C:\VPTR_jigsaws\Suturing\Suturing\video'
    meta_file_path = r'C:\VPTR_jigsaws\Suturing\Suturing\meta_file_Suturing.txt'
    transcriptions_directory = r'C:\VPTR_jigsaws\Suturing\Suturing\transcriptions'
    output_split_dir = r'C:\VPTR_jigsaws\jigsaws_suturing\frames_split'
    preprocess_and_split_dataset(videos_directory, meta_file_path, transcriptions_directory, output_split_dir)