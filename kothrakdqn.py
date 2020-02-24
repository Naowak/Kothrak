import sys
import Kothrak as Kk
import torch
import threading
import queue
import random
import math
from collections import namedtuple
# if gpu is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from itertools import count
import time

from PyQt5.QtWidgets import QApplication



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


"""## Neural Networks Model"""

# Pure copy/paste of
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Should be improved later
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(23, 6, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = F.relu(x)
        return x        

def select_action(env):
    global steps_done
    env.create_masks()

    state = env.state()
    # current_player = env.get_current_player_index()
    mask = env.move_mask
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            pn = policy_net(torch.tensor(state, device=DEVICE, dtype=torch.long))
            pn = torch.tensor(env.apply_mask(pn, mask), device=DEVICE, dtype=torch.long)





            # max(1)[1].view(1, 1)
            # return policy_net(state)
            return pn.argmax(keepdim=True), torch.tensor(state, device=DEVICE, dtype=torch.long)
    else:
        pn = torch.tensor([[random.randrange(n_actions)]], device=DEVICE, dtype=torch.long)
        pn = torch.tensor(env.apply_mask(pn, mask), device=DEVICE, dtype=torch.long)
        return pn.argmax(), torch.tensor(state, device=DEVICE, dtype=torch.long)

def transform_action(action) :
    coord_actions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
    return coord_actions[action]


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=DEVICE, dtype=torch.long)

    # tmp = torch.tensor([s for s in batch.next_state if s is not None], device=DEVICE, dtype=torch.long)
    # tmp = tuple(torch.tensor(s, device=DEVICE, dtype=torch.long) for s in batch.next_state if s is not None)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    # non_final_next_states = torch.cat(tmp)
    state_batch = torch.cat(batch.state).view(23, -1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_game_state(env) :
    return torch.tensor(env.state(), device=DEVICE, dtype=torch.long)


def run_game(queue) :
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(Kk.style)
    env = Kk.MyApp()
    q.put(env)
    env.show()
    sys.exit(qapp.exec_())


q = queue.Queue()
t = threading.Thread(target=run_game, args=(q,))
t.deamon = True
t.start()
env = q.get()

# replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get number of actions from gym action space
n_actions = 6

target_net = DQN().to(DEVICE)
policy_net = DQN().to(DEVICE)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

episode_durations = []


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    # env.new_game()
    for t in count():
        # Select and perform an action
        action, last_state = select_action(env)
        print(i_episode, 'try')
        reward, done = env.play(*transform_action(action.item()))
        reward = torch.tensor(reward, device=DEVICE, dtype=torch.long)

        # Observe new state
        state = torch.tensor(env.state(), device=DEVICE, dtype=torch.long)

        # Store the transition in memory
        memory.push(last_state, action.view(-1, 1), state, reward.view(-1, 1))

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print   ('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

