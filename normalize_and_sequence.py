import cv2
import os
import numpy as np


def normalize_and_generate_sequences(input_dir, output_dir, target_size=(224, 224), sequence_length=10):
    """
    Normalize frames and generate sequences.
    Args:
        input_dir (str): Path to the directory containing resized frames.
        output_dir (str): Path to save the generated sequences.
        target_size (tuple): Target size for resizing (width, height).
        sequence_length (int): Number of frames per sequence.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Loop through splits (train, test, validate)
    for split in os.listdir(input_dir):
        split_path = os.path.join(input_dir, split)
        split_output_path = os.path.join(output_dir, split)
        os.makedirs(split_output_path, exist_ok=True)

        # Loop through classes (fight, non_fight)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            class_output_path = os.path.join(split_output_path, class_name)
            os.makedirs(class_output_path, exist_ok=True)

            # Loop through videos (video1, video2, ...)
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                frames = sorted(os.listdir(video_path))
                video_output_path = os.path.join(class_output_path, video_folder)
                os.makedirs(video_output_path, exist_ok=True)

                normalized_frames = []

                # Normalize all frames
                for frame_name in frames:
                    frame_path = os.path.join(video_path, frame_name)
                    frame = cv2.imread(frame_path)
                    frame = cv2.resize(frame, target_size)  # Resize frame

                    # Normalize pixel values to range [0, 1]
                    normalized_frame = frame / 255.0
                    normalized_frames.append(normalized_frame)

                # Generate sequences
                for i in range(0, len(normalized_frames) - sequence_length + 1, sequence_length):
                    sequence = np.array(normalized_frames[i:i + sequence_length])

                    # Save sequence as numpy array
                    sequence_path = os.path.join(video_output_path, f"sequence_{i}.npy")
                    print(f"Saving sequence to {sequence_path}")
                    np.save(sequence_path, sequence)

    print("Sequences generated and saved successfully.")


# Example usage
normalize_and_generate_sequences(
    input_dir="resized_dataset_frames",
    output_dir="dataset_sequences",
    target_size=(224, 224),
    sequence_length=10
)
