from flask import Flask, request, jsonify
from model import Model
from utils import extract_frames

app = Flask(__name__)
model = Model()

@app.route('/predict', methods=['POST'])
def predict():
    # Expecting a video file in the POST request
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Save the uploaded video file
    video_path = './data/uploaded_video.mp4'
    file.save(video_path)

    # Process the video to extract violence predictions
    predictions = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label = model.predict(frame)['label']
        predictions.append(label)
    cap.release()
    
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
