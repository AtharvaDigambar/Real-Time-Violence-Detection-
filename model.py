import clip
import cv2
import numpy as np
import torch
import yaml
from PIL import Image

class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.labels = self.settings['label-settings']['labels']
        
        # Adjusted labels for better CLIP accuracy
        self.labels_ = ["a photo of " + label for label in self.labels]
        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings'].get('default', 'non-violence')
    
    def vectorize_text(self, texts):
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features

    def predict(self, image):
        image_input = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            logits_per_image = (image_features @ self.text_features.T).softmax(dim=-1)
            probs, idx = logits_per_image.max(dim=1)
            label = self.labels[idx] if probs.item() > self.threshold else self.default_label

        return {'label': label, 'confidence': probs.item()}
