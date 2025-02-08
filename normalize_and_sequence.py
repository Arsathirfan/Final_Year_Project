import cv2
import os
import numpy as np


def normalize_and_generate_sequences(input_dir, output_dir, target_size=(224, 224), sequence_length=10):

    os.makedirs(output_dir, exist_ok=True)

    # Loop through classes (fight, no_fight)
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)

        if os.path.isdir(class_path):  # Only process directories
            class_output_path = os.path.join(output_dir, class_name)
            os.makedirs(class_output_path, exist_ok=True)

            # Loop through videos (fi001, fi002, ...)
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)

                if os.path.isdir(video_path):  # Only process directories (video folders)
                    video_output_path = os.path.join(class_output_path, video_folder)
                    os.makedirs(video_output_path, exist_ok=True)

                    frames = sorted(os.listdir(video_path))  # Sort frames for proper sequence generation
                    normalized_frames = []

                    # Normalize all frames
                    for frame_name in frames:
                        frame_path = os.path.join(video_path, frame_name)
                        frame = cv2.imread(frame_path)

                        if frame is not None:
                            frame = cv2.resize(frame, target_size)  # Resize frame

                            # Normalize pixel values to range [0, 1]
                            normalized_frame = frame / 255.0
                            normalized_frames.append(normalized_frame)
                        else:
                            print(f"Warning: Failed to read {frame_path}. Skipping.")

                    # Generate sequences and save as numpy arrays
                    for i in range(0, len(normalized_frames) - sequence_length + 1, sequence_length):
                        sequence = np.array(normalized_frames[i:i + sequence_length])

                        # Save sequence as numpy array
                        sequence_path = os.path.join(video_output_path, f"sequence_{i}.npy")
                        print(f"Saving sequence to {sequence_path}")
                        np.save(sequence_path, sequence)

    print("Sequences generated and saved successfully.")


# Example usage
normalize_and_generate_sequences(
    input_dir="D:\\project_folder\\resized_dataset_frames",
    output_dir="D:\\project_folder\\normalized_sequenced_frames",
    target_size=(299, 299),
    sequence_length=10
)
