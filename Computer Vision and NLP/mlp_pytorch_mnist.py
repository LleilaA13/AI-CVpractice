import torch 
from torch.utils import Dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics

#load the training and test datasets
training_data = datasets.FashionMNIST(train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(train=False, download=True, transform=ToTensor())

#get the best device for computation
device = ('cuda' if torch.cuda.is_available() else 'cpu')

#create our MLP model
class OurMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass