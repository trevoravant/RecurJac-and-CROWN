import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

net_name = 'compnet'
plot_name = 'Comparison Net'

# input size
#n_input = 3
n_input = 7
n_output = 10
numlayer = 3

# set random seed for reproducibility (for both torch.rand and torch.nn.Linear)
torch.manual_seed(0)

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        #self.fc1 = nn.Linear(n_input,30,bias=True)
        #self.fc2 = nn.Linear(30,50,bias=True)
        #self.fc3 = nn.Linear(50,20,bias=True)
        #self.fc4 = nn.Linear(20,10,bias=True)
        self.fc1 = nn.Linear(n_input,20,bias=True)
        self.fc2 = nn.Linear(20,30,bias=True)
        self.fc3 = nn.Linear(30,n_output,bias=True)
        #self.fc4 = nn.Linear(20,10,bias=True)

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc4(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# nominal input
#x0 = torch.rand(1,n_input)
x0 = torch.zeros(1,n_input)

def net():
    net = MyNet()

    relu = torch.nn.ReLU(inplace=False)
    #net.layers = [net.fc1, relu,
                  #net.fc2, relu,
                  #net.fc3, relu,
                  #net.fc4]
    net.layers = [net.fc1, relu,
                  net.fc2, relu,
                  net.fc3]

    return net
