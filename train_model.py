import numpy
import torch
import os
import pandas
from torch import nn
from loadNumbers import LoadImageDataset, NeuralNetwork
from torch.utils.data import DataLoader
from torchvision import transforms


ANNOTATIONS = 'labels.csv'
IMG_PATH = 'Images'
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


dataset = LoadImageDataset(ANNOTATIONS, IMG_PATH, transform=transform)





train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])

train_load = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_load = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

for X, y in train_load:
    print("shape:", X.shape)
    print("dtype:", X.dtype)
    print("min/max:", X.min().item(), X.max().item())
    break

loss_fn = nn.CrossEntropyLoss()

model = NeuralNetwork()




optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_load, model, loss_fn, optimizer)
    test_loop(test_load, model, loss_fn)

PATH = 'model.pth'
torch.save(model.state_dict(), PATH)

print("Done!")

