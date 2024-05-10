import torch 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics
import matplotlib.pyplot as plt

#load the training and test datasets
training_data = datasets.FashionMNIST(root = 'data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root = 'data', train=False, download=True, transform=ToTensor())
img, label = training_data[25]
plt.imshow(img.squeeze(), cmap='gray') #squeeze removes the channel dimension, if we provide a 3 dimensional tensor, 
#we will have to remove the channel dimension
plt.show()
#get the best device for computation
device = ('cuda' if torch.cuda.is_available() else 'cpu')

#create our MLP model
class OurMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(28*28, 50),
            nn.Sigmoid(),
            nn.Linear(50, 100),
            nn.Sigmoid(),
            nn.Linear(100, 50),
            nn.Sigmoid(),
            nn.Linear(50, 10),
            
        )
        self.flatten = nn.Flatten() #flatten the input tensor, convert into a mono dimensional array in order to feed it into the MLP
    def forward(self, x): #x is a sample from the dataset, we need to convert our inpot tensor into a mono dimensional array
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits


# instantiate the model
model = OurMLP().to(device)

#define the hyperparameters
epochs = 2
batch_size = 16
learning_rate = 0.001

#define the loss function
loss_function = nn.CrossEntropyLoss()

#todo: define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#todo: create the dataloader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#define the accuracy metric
accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = 10)

#define the training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        #compute the prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        '''if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')'''
