from collections import namedtuple
import random
from time import sleep
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepQNetwork(nn.Module):

    def __init__(self, num_input, num_output):

        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(num_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, num_output)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


def Trainer():

    TIME_TO_SLEEP = 0

    def __init__(self, name, num_inputs, num_actions, lr, gamma, batch_size, 
            size_min_memory, size_max_memory, hidden_units):
        self.name = name
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.size_min_memory = size_min_memory
        self.size_max_memory = size_max_memory
        self.hidden_units = hidden_units

        self.policy_net = DeepQNetwork(num_inputs)
        self.target_net = DeepQNetwork(num_inputs)
        self.update_TargerNet()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []


    def run(self):
        pass

    def run_one_game(self):
        """Play one game and optimize model."""
        sum_reward = 0
        done = False
        state = torch.from_numpy(self.env.reset())
        losses = list()

        while not done:

            # Choose action in function of observation and play it
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action.item())

            sum_reward += reward
            next_state = torch.from_numpy(next_state)
            reward = torch.tensor([reward], device=device)
            
            # Add transition to memory
            self.add_to_memory(state, action, next_state, reward, done)

            # Compute loss
            loss = self.optimize_model()
            losses += [loss]
            
            # Update graphic events
            self.qapp.processEvents()
            sleep(TIME_TO_SLEEP)
            
            # Prepare next state
            state = next_state

        return sum_reward, mean(losses)


    def optimize_model(self):
        """Train the model by selecting a random subset of combinaison
        (state, action) in his last experiences, calcul the loss from q_values,
        and apply back-propagation."""

        # Check that there is enough plays in self.experiences
        if len(self.memory) < self.size_min_memory:
            return 0

        # Select self.batch_size random experience
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        # Compute Q(s, a) for all state
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q(s_{t+1}) for all next_state
        next_q_values = torch.zeros(self.batch_size, device=device)
        next_q_values[~done_batch] = self.target_net(
            next_state_batch[~done_batch]).max(1)[0].detach()

        # Compute expected Q-value
        expected_q_values = (next_q_values * self.gamma) + reward_batch

        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


    def select_action(self, state):
        """Choose randomly in function of epsilon between a random action
        or the action having the best q_value."""
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
            return torch.tensor([[action]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)


    def add_to_memory(self, state, action, next_state, reward, done):
        """Add a new transition to the memory, remove the first ones if there
        is no more room in the memory."""
        if len(self.memory) >= self.size_max_memory:
            self.memory.pop(0)
        self.memory += [Transition(state, action, next_state, reward, done)]


    def update_TargerNet(self):
        """Copy the weights and biais from policy_net to target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
