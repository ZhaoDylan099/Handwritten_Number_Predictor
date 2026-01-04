import torch
from interface import DrawApp
from loadNumbers import MLP, CNNModel




MODEL_PATH = "CNN.pth"

if MODEL_PATH.endswith('MLP.pth'):
    model = MLP()
else:
    model = CNNModel()

model.load_state_dict(torch.load(MODEL_PATH))

DrawApp(model)
