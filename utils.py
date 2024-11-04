import os
import cv2

def extract_frames(video_path, output_folder, frame_rate=1):
    """Extract frames from a video at a specified frame rate."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0
    success = True

    os.makedirs(output_folder, exist_ok=True)

    while success:
        success, frame = cap.read()
        if not success:
            break
        
        # Save frames at a specific rate
        if count % frame_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")
