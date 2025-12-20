import torch
from interface import DrawApp
from loadNumbers import NeuralNetwork




MODEL_PATH = "model.pth"

model = NeuralNetwork()

model.load_state_dict(torch.load(MODEL_PATH))

DrawApp(model)
