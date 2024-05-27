import torch
from model import load_model

# Load the pre-trained model
model = load_model('model_weights.pth')

def predict_diabetes(data):
    with torch.no_grad():
        return model(data)
