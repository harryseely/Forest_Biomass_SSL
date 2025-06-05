import torch
import torch.nn as nn

class Regressor(torch.nn.Module):

    def __init__(self, in_channels, num_outputs, n_neurons=64):
        super().__init__()

        activation_fn = nn.LeakyReLU()

        self.regressor = nn.Sequential(
            nn.Linear(in_channels, n_neurons),
            activation_fn,
            nn.Linear(n_neurons, num_outputs))
    
    def forward(self, x):
        
        x = self.regressor(x)
        x = x.squeeze()

        return x
    
