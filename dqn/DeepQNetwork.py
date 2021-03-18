import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):

    def __init__(self, num_input, hidden_layers, num_output):
        """Initialize the dqn architecture.
        - num_input: number of observations
        - hidden_layer: list containing a number of neurons for each layer
        - num_output: number of actions
        """
        # def init_layer(layer):
        #     nn.init.normal_(layer.bias)
        #     nn.init.normal_(layer.weight)

        super(DeepQNetwork, self).__init__()

        # # Create as many layers as contained in hidden_layers
        # for i, num_neurons in enumerate(hidden_layers + [num_output]):
        #     layer = nn.Linear(num_input, num_neurons)
        #     init_layer(layer)
        #     print('bias', layer.bias.size())
        #     print('weight', layer.weight.size())
        #     setattr(self, f'fc{i+1}', layer)
        #     num_input = num_neurons
        import pickle
        import torch

        dico = pickle.load(open('params.pick', 'rb'))
        self.fc1 = nn.Linear(40, 150)
        self.fc2 = nn.Linear(150, 6)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(dico['weight1']).float())
            self.fc1.bias.copy_(torch.from_numpy(dico['bias1']).float())
            self.fc2.weight.copy_(torch.from_numpy(dico['weight2']).float())
            self.fc2.bias.copy_(torch.from_numpy(dico['bias2']).float())

    def forward(self, x):
        """Make a forward pass and return the results (predictions).
        - x : input data
        """
        layers = [getattr(self, k) for k in self.__dict__['_modules'] if 'fc' in k]
        
        for layer in layers[:-1]:
            x = F.relu(layer(x))
        x = layers[-1](x)
        
        return x
