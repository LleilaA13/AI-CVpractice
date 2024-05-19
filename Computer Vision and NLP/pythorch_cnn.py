"""
Created Mer 15/05/24 - 11:36 - AI lab - continued on Ven 17/05/2024 - 14:44
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


# define our CNN
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3),
            # nn.BatchNorm2d(),  - normalizes output
            nn.ReLU(),
            nn.Conv2d(5, 10, 3),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(24 * 24 * 10, 10),
            # nn.Dropout(0.5),  - disconnects edges between neurons, preventing overfitting
            nn.ReLU(),
            nn.Linear(10, 10)

        )

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.mlp(x)
        return x


# instance of the model
model = OurCNN().to(device)

# test_x = torch.rand((1, 28, 28))
# test_y = model(test_x)
#
# exit()

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

    # disable the weight update:
    # model.eval()  - batch norm layers are removed from the model
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
