import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):

    def __init__(self, num_input, hidden_layers, num_output):
        """Initialize the dqn architecture.
        - num_input: number of observations
        - hidden_layer: list containing a number of neurons for each layer
        - num_output: number of actions
        """
        super(DeepQNetwork, self).__init__()

        # Create as many layers as contained in hidden_layers
        for i, num_neurons in enumerate(hidden_layers + [num_output]):
            setattr(self, f'fc{i+1}', nn.Linear(num_input, num_neurons))
            num_input = num_neurons

    def forward(self, x):
        """Make a forward pass and return the results (predictions).
        - x : input data
        """
        layers = [getattr(self, k) for k in self.__dict__['_modules'] if 'fc' in k]
        
        for layer in layers[:-1]:
            x = F.relu(layer(x))
        x = layers[-1](x)
        
        return x
