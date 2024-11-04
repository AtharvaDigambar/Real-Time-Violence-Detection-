# Real-Time-Violence-Detection-
This project is an AI-powered violence detection system that analyzes video frames in real time to classify them as "violence" or "non-violence." Leveraging OpenAI’s CLIP model, the system processes video input to detect signs of violent behavior in public spaces, enhancing security and safety. 
# Violence Detection in Videos

This project uses AI to detect violence in videos in real-time. It utilizes OpenAI's CLIP model to classify frames as either "violence" or "non-violence."

## Files
- `run.py`: Main script to run violence detection on video files.
- `model.py`: Contains the Model class using the CLIP model.
- `utils.py`: Utility functions for tasks such as frame extraction.
- `app.py`: Flask app for serving the model as an API.
- `tutorial.ipynb`: Jupyter notebook tutorial for understanding the project.
- `settings.yaml`: Config file for model settings and labels.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/violence-detection.git
cd violence-detection
