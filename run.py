import argparse
import cv2
import os
from model import Model
from utils import extract_frames

def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection in videos")
    parser.add_argument('C:\Users\OneDrive\Desktop\Videos', type=str,
                        default=r'C:\Users\OneDrive\Desktop\Videos')
    args = parser.parse_args()
    return args

def process_videos_in_folder(video_folder, model):
    # Iterate over all video files in the specified folder
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        if not os.path.isfile(video_path):
            continue
        
        print(f"Processing video: {video_file}")
        process_video(video_path, model)

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run prediction on the current frame
        label = model.predict(image=frame)['label']
        
        # Display the frame with label overlay
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Violence Detection', frame)
        
        # Press 'q' to exit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argument_parser()
    model = Model()
    process_videos_in_folder(args.video_folder, model)
