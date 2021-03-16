import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):

    def __init__(self, num_input, hidden_layers, num_output):

        super(DeepQNetwork, self).__init__()

        for i, num_neurons in enumerate(hidden_layers + [num_output]):
            setattr(self, f'fc{i+1}', nn.Linear(num_input, num_neurons))
            num_input = num_neurons

    def forward(self, x):
        
        layers = [getattr(self, k) for k in self.__dict__ if 'fc' in k]
        
        for layer in layers[:-1]:
            x = F.relu(layer(x))
        x = layer[-1](x)
        
        return x
