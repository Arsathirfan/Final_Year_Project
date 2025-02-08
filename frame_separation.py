import os
import cv2

def extract_frames_from_videos(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if not os.path.isdir(category_path):
            continue

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            video_name, _ = os.path.splitext(video_file)

            if not os.path.isfile(video_path):
                continue

            # Create directory for the current video's frames
            video_frames_path = os.path.join(output_category_path, video_name)
            if not os.path.exists(video_frames_path):
                os.makedirs(video_frames_path)

            # Read the video
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_file_name = f"frame_{frame_count:04d}.jpg"
                frame_file_path = os.path.join(video_frames_path, frame_file_name)

                # Save the frame as an image
                cv2.imwrite(frame_file_path, frame)
                frame_count += 1

            cap.release()
            print(f"Extracted {frame_count} frames from {video_file}.")

if __name__ == "__main__":
    input_directory = "D:\\project_folder\\data_set"
    output_directory = "D:\\project_folder\\data_set_frames"
    extract_frames_from_videos(input_directory, output_directory)
