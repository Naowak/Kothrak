import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Net(nn.Module):

    def __init__(self, num_input, hidden_neurons, num_output):

        super(Net, self).__init__()

        # self.layers = []
        # input_features = num_input

        # for out_features in hidden_neurons:
        #     self.layers += [nn.Linear(input_features, out_features)]
        #     input_features = out_features

        # self.output_layer = nn.Linear(input_features, num_output)

        self.fc1 = nn.Linear(40, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 6)

    def forward(self, x):
        
        # for layer in self.layers:
        #     x = layer(x)
        #     x = F.relu(x)
        
        # x = self.output_layer(x)

        # output = F.log_softmax(x, dim=1)
        # return output

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Net(40, [250, 100], 6).to(device)
summary(net, (1, 40))
# params = print(list(net.parameters()))
# print(len(params))
# print(params[0].size())

random_data = torch.rand(1, 40).to(device)
output = net(random_data)

# net.zero_grad()
# output.backward(torch.randn(1, 6))

target = torch.randn(6).to(device)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


net.zero_grad()     # zeroes the gradient buffers of all parameters
print('fc1.bias.grad before backward')
print(net.fc1.bias.grad)

loss.backward()
print('fc1.bias.grad after backward')
print(net.fc1.bias.grad)


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # Does the update


print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
