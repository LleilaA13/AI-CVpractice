"""
Created Mer 15/05/24 - 10:25 - AI lab
author@ walver
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics
import matplotlib.pyplot as plt

# load the training data
training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

# img, label = training_data[25]
# plt.imshow(img.squeeze(), cmap='gray')
# plt.show()

# get the best device for computation
device = ('cuda' if torch.cuda.is_available() else 'cpu')


# create MLP
class OurMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(28 * 28, 50)
        self.hidden1 = nn.Linear(50, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.output_layer = nn.Linear(50, 100)
        self.activation = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        logits = self.output_layer(x)

        return logits


# instance of the model
model = OurMLP().to(device)

# define the hyperparameters
epochs = 2
batch_size = 16
learning_rate = 0.001

# define the optimizer
loss_fn = nn.CrossEntropyLoss()

# define the optimizer:
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# create the dataloader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# define the accuracy metric
metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)


# define the training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)

    # get the batch from the dataset
    for batch, (X, y) in enumerate(dataloader):

        # move data to the device
        X = X.to(device)
        y = y.to(device)

        # compute the prediction and the loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # let's adjust the weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print some information
        if batch % 20 == 0:
            loss_v, current_batch = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss_v} [{current_batch}/{size}]')
            acc = metric(pred, y)
            print(f'Accuracy on current batch: {acc}')

        # print the final accuracy
        acc = metric.compute()
        print(f'final accuracy: {acc}')
        metric.reset()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader)

    # disable the weight update
    with torch.no_grad():
        for X, y in dataloader:
            # move data to the correct devise
            X = X.to(device)
            y = y.to(device)

            # get the model predictions
            pred = model(X)

            # compute the accuracy
            acc = metric(pred, y)

    # compute the final accuracy
    acc = metric.compute()

    print(f'final testing accuracy: {acc}')
    metric.reset()


# train the model
for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print('Done')
