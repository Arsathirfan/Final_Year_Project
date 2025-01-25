import cv2
import os


def resize_frames(input_dir, output_dir, target_size=(224, 224)):
    """
    Resize all frames in the dataset to a consistent size.
    Args:
        input_dir (str): Path to the directory containing the frames.
        output_dir (str): Path to save the resized frames.
        target_size (tuple): Target size for resizing (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)

    for split in os.listdir(input_dir):  # train, test, validate
        split_path = os.path.join(input_dir, split)
        split_output_path = os.path.join(output_dir, split)
        os.makedirs(split_output_path, exist_ok=True)

        for class_name in os.listdir(split_path):  # fight, non_fight
            class_path = os.path.join(split_path, class_name)
            class_output_path = os.path.join(split_output_path, class_name)
            os.makedirs(class_output_path, exist_ok=True)

            for video_folder in os.listdir(class_path):  # video1, video2
                video_path = os.path.join(class_path, video_folder)
                video_output_path = os.path.join(class_output_path, video_folder)
                os.makedirs(video_output_path, exist_ok=True)

                for frame_name in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame_name)
                    output_frame_path = os.path.join(video_output_path, frame_name)

                    # Read, resize, and save the frame
                    frame = cv2.imread(frame_path)
                    resized_frame = cv2.resize(frame, target_size)
                    cv2.imwrite(output_frame_path, resized_frame)
                    print(f"Resized {frame_name}.")
    print("Resizing complete.")


# Example usage
resize_frames("dataset_frames", "resized_dataset_frames", target_size=(224, 224))
