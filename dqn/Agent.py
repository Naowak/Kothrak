
import os
import pickle
import shutil
import random
from statistics import mean
from collections import namedtuple
from datetime import datetime

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from dqn.DeepQNetwork import DeepQNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

CHANGEABLE_PARAMETERS = ['name', 'update_frequency', 'epsilon', 'decay', 
                'min_epsilon', 'lr', 'gamma', 'batch_size', 'memory_size']

CONSTANT_PARAMETERS = ['num_observations', 'num_actions', 'hidden_layers'] 

HIDDEN_PARAMETERS = ['_game_played', '_memory']


class Agent():

    def __init__(self, num_observations, num_actions, hidden_layers=[50, 50], 
                    name='', update_frequency=50, epsilon=0.99, decay=0.9999, 
                    min_epsilon=0, lr=0.001, gamma=0.99, batch_size=32,
                    memory_size=10000):

        # Constant attributes
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.hidden_layers = hidden_layers

        # Changeable attributes
        self.name = name if name != '' else \
                                    datetime.now().strftime("%m%d%y-%H%M")
        self.update_frequency = update_frequency
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        # Hidden attributes
        self._game_played = 0
        self._memory = []

        # SummaryWriter for Logs
        self._summary_writer = None
        self._reset_summary_writer(self._summary_writer)

        # Torch instances
        self._policy_net = None
        self._target_net = None
        self._optimizer = None
        self._create_neural_networks()
        self._create_optimizer()

        # Temporary attributes
        self._rewards = []
        self._losses = []
        self._last_rewards = []
        self._last_losses = []

        # Display
        summary(self._policy_net, (1, self.num_observations))


    def play(self, state, mask, train=False): 
        """If train is true, choose randomly in function of epsilon between a 
        random action or the action having the best q_value.
        If train is false, return the action with the best q-value.
        - state : list of values describing the game
        - train : boolean
        """
        if 1 not in mask:
            print('Nope', mask)
            # No action possible, make an action to get eliminated
            return torch.tensor([[0]])

        if train and random.random() < self.epsilon:
            # Random action
            action = random.choice([i for i, v in enumerate(mask) if v])
            action = torch.tensor([[action]], device=device, dtype=torch.long)         
        else:
            # Best action
            with torch.no_grad():
                mask = [v == 1 for v in mask]
                mask = torch.tensor(mask, device=device)
                infs = torch.full((1, 36), float('-inf'), device=device)
                preds = self._policy_net(state)
                preds = torch.where(mask, preds, infs)
                action = preds.max(1)[1].view(1, 1)
                print(mask)
                print(preds)
                print(action)
        return action


    def update(self, state, action, next_state, reward, done):
        """Update memory, compute loss, optimize_model and update target_net 
        and epsilon parameters at each update_frequency game played.
        """
        # Convert to tensor
        # state = torch.tensor(state, device=device).view(1, -1)
        # action = torch.tensor([action])
        next_state = torch.tensor(next_state, device=device).view(1, -1)
        reward = torch.tensor([reward], device=device)
        done = torch.tensor([done], device=device)

        # Optimize
        self._add_to_memory(state, action, next_state, reward, done)
        loss = self._optimize_model()

        # Logs memory
        self._rewards += [reward.item()]
        self._losses += [loss]

        # Game over
        if done:

            self._game_played += 1

            # Epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

            # Reward & Loss
            game_reward = sum(self._rewards)
            game_loss = mean(self._losses)
            self._rewards = []
            self._losses = []
            self._last_rewards += [game_reward]
            self._last_losses += [game_loss]

            # Logs
            self._summary_writer.add_scalar('Reward', game_reward, 
                self._game_played)
            self._summary_writer.add_scalar('Loss', game_loss, 
                self._game_played)
            self._summary_writer.add_scalar('Epsilon', self.epsilon, 
                self._game_played)

            # Each update_frequency games
            if self._game_played % self.update_frequency == 0:

                # Copy weights 
                self._update_target_net()

                # Display
                print(f'{self.name}: ',
                    f'Game played: {self._game_played}, '
                    f'Epsilon: {self.epsilon}, '
                    f'Mean reward: {mean(self._last_rewards)}, '
                    f'Mean loss: {mean(self._last_losses)} ')

                # Reset after display
                self._last_rewards = []
                self._last_losses = []


    def set_parameters(self, **parameters):
        """Set all couple (k, v) in **parameters as self.k = v if k is in 
        CHANGEABLE_PARAMETERS. If lr is changed, create a new optimizer.
        If name is changed, create new summary_writer.
        - Any k=v couple values, where k is the name of the parameter, and
        v the value.
        """
        for param, value in parameters.items():
            
            # Param valid handle
            if param not in CHANGEABLE_PARAMETERS:     
                raise Exception(f'Parameter {param} not known.')
            
            # Flag handle
            if getattr(self, param) != value:

                if param == 'name':
                    # if name is modified, path to the summary_writer change
                    self._reset_summary_writer(self.name)

                if param == 'lr':
                    self._create_optimizer(value)

            # Set value
            setattr(self, param, value)


    def get_parameters(self):
        """Return all changeable parameters values.
        """
        params = {}
        for p in CHANGEABLE_PARAMETERS:
            params[p] = getattr(self, p)
        return params


    def _optimize_model(self):
        """Train the model by selecting a random subset of combinaison
        (state, action) in his memory, calcul the loss from q_values,
        and apply back-propagation.
        """
        # Check that there is enough plays in self.experiences
        if len(self._memory) < self.batch_size:
            return 0

        # Select self.batch_size random experience
        transitions = random.sample(self._memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        # Compute Q(s, a) for all state
        q_values = self._policy_net(state_batch).gather(1, action_batch)

        # Compute Q(s_{t+1}) for all next_state
        next_q_values = torch.zeros(self.batch_size, device=device)
        if sum(~done_batch) != 0:
            with torch.no_grad():
                next_q_values[~done_batch] = self._target_net(
                    next_state_batch[~done_batch]).max(1)[0].detach()

        # Compute expected Q-value
        expected_q_values = (next_q_values * self.gamma) + reward_batch

        # Compute loss
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()


    def _add_to_memory(self, state, action, next_state, reward, done):
        """Add a new transition to the memory, remove the first ones if there
        is no more room in the memory.
        - state : list of values describing the game
        - action : a number representing an action
        - next_state : list of values describing the game after the action
        - reward : value rewarding the player for his action
        - done : Boolean indicating if the game is finished of not
        """
        if len(self._memory) >= self.memory_size:
            self._memory.pop(0)
        self._memory += [Transition(state, action, next_state, reward, done)]


    def _reset_summary_writer(self, old_name):
        """Create a new folder at ./logs/name/. If no data were written in
        the old folder, delete it.
        """
        if old_name is not None:
            
            dirname = f'./logs/{old_name}/'
            
            if os.path.exists(dirname):
                filename = os.listdir(dirname)[0]

                # If no data in old dirname, delete directory
                if os.stat(dirname + filename).st_size == 40:
                    shutil.rmtree(dirname)

        self._summary_writer = SummaryWriter(log_dir=f'./logs/{self.name}/')


    def _update_target_net(self):
        """Copy the weights and biais from policy_net to target_net
        """
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()


    def _create_neural_networks(self):
        """Create policy_net and target_net, and copy weights from policy_net
        to target_net.
        """
        self._policy_net = DeepQNetwork(self.num_observations,
                                        self.hidden_layers, 
                                        self.num_actions).to(device)
        self._target_net = DeepQNetwork(self.num_observations,
                                        self.hidden_layers, 
                                        self.num_actions).to(device)
        self._update_target_net()


    def _create_optimizer(self, learning_rate=None):
        """Create optimizer with lr. Neural network has to be create before
        calling this method.
        """        
        if learning_rate is None:
            learning_rate = self.lr
        self._optimizer = optim.Adam(self._policy_net.parameters(), 
            lr=learning_rate, eps=1e-7)


def save_agent(agent, directory='saves/'):
    """Save the agent's model, optimizer and the trainer parameters.
    """
    # Create dirpath for temporary dir
    if directory[-1] != '/':
        directory += '/'
    dirpath = directory + agent.name + '/'

    if not os.path.exists(dirpath): 
        os.makedirs(dirpath)
    else:
        raise Exception(f'Path {dirpath} already exists.')

    # DQNs & Optimizer
    torch.save(agent._policy_net.state_dict(), f'{dirpath}dqn.pth')
    torch.save(agent._optimizer.state_dict(), f'{dirpath}optimizer.pth')

    # Trainer pamameters
    params = {'modifiable': {}, 'static': {}, 'hidden': {}}

    for p in CHANGEABLE_PARAMETERS:
        params['modifiable'][p] = getattr(agent, p)

    for p in CONSTANT_PARAMETERS:
        params['static'][p] = getattr(agent, p)

    for p in HIDDEN_PARAMETERS:
        params['hidden'][p] = getattr(agent, p)

    with open(f'{dirpath}parameters.pick', 'wb') as file:
        pickle.dump(params, file)

    # Zip the saves in one .zip archive
    zippath = f'{directory}{agent.name}'
    shutil.make_archive(zippath, 'zip', dirpath)

    # Remove the directory dirpath and files inside
    shutil.rmtree(dirpath)

    # Display
    print(f'Agent saved at {zippath}.zip')


def load_agent(filename, directory_tmp='saves/tmp/'):
    """Load the model, optimizer and trainer parameters and return 
    the corresponding Agent.
    """
    # Verify path
    if not os.path.exists(filename):
        raise IOError(f'Filename {filename} does not exists.')

    if os.path.exists(directory_tmp):
        raise Exception(f'Path {directory_tmp} already exists, \
            please choose a non-existant path.')

    if directory_tmp[-1] != '/':
        directory_tmp += '/'

    # Unzip the archive
    shutil.unpack_archive(filename, directory_tmp, 'zip')
    
    # Trainer parameters
    with open(f'{directory_tmp}parameters.pick', 'rb') as file:
        params = pickle.load(file)

    # Create instance with params
    agent = Agent(**params['static'], **params['modifiable'])
    agent._game_played = params['hidden']['_game_played']
    agent._memory = params['hidden']['_memory']

    # DQNs & Optimizer
    agent._policy_net.load_state_dict(torch.load(f'{directory_tmp}dqn.pth'))
    agent._policy_net.eval()

    agent._update_target_net()
    agent._target_net.eval()

    agent._optimizer.load_state_dict(
        torch.load(f'{directory_tmp}optimizer.pth'))

    # Remove the directory directory_tmp and files inside
    shutil.rmtree(directory_tmp)

    # Display
    print(f'Agent {agent.name} loaded from {filename}.')

    return agent
