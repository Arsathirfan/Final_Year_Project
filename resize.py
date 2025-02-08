import cv2
import os


def resize_frames(input_dir, output_dir, target_size=(224, 224)):

    # Iterate over the top-level directories (fight, no_fight)
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)

        if os.path.isdir(class_path):  # Only process directories (like 'fight' and 'no_fight')
            class_output_path = os.path.join(output_dir, class_name)
            os.makedirs(class_output_path, exist_ok=True)

            # Iterate over subfolders (fi001, fi002, etc.)
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)

                if os.path.isdir(video_path):  # Only process directories (like 'fi001')
                    video_output_path = os.path.join(class_output_path, video_folder)
                    os.makedirs(video_output_path, exist_ok=True)

                    # Iterate over the frames in the video folder (frame_0000.jpg, etc.)
                    for frame_name in os.listdir(video_path):
                        frame_path = os.path.join(video_path, frame_name)

                        if os.path.isfile(frame_path):  # Only process image files
                            output_frame_path = os.path.join(video_output_path, frame_name)

                            # Read the frame
                            frame = cv2.imread(frame_path)

                            if frame is not None:  # Check if image is loaded properly
                                print(f"Resizing {frame_name}...")  # Debugging log
                                resized_frame = cv2.resize(frame, target_size)

                                # Save the resized frame
                                cv2.imwrite(output_frame_path, resized_frame)
                                print(f"Resized and saved: {output_frame_path}")
                            else:
                                print(f"Failed to read {frame_name}. Skipping.")

    print("Resizing complete.")


# Example usage
resize_frames("D:\\project_folder\\data_set_frames", "D:\\project_folder\\resized_dataset_frames",
              target_size=(299, 299))
