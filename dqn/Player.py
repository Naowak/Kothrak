
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

pid = 0


class Player():

    PARAMETERS = ['name', 'update_frequency',
                    'epsilon', 'decay', 'min_epsilon', 'lr', 'gamma', 
                    'batch_size', 'hidden_layers']

    _DEFAULT_VALUES = {'name': datetime.now().strftime("%m%d%y-%H%M"), 
                        'update_frequency': 20,
                        'epsilon': 0.99, 
                        'decay': 0.9998,
                        'min_epsilon': 0.01, 
                        'lr': 1e-3, 
                        'gamma': 0.99,
                        'batch_size': 32,
                        'hidden_layers': [150],
                        'game_played': 0, 
                        'memory': []}

    def __init__(self, num_observations, num_actions):
        
        # Definitive attributes
        self.num_inputs = num_observations
        self.num_outputs = num_actions
        self.size_max_memory = 2000

        # Parameters (values can change)
        self.name = None
        self.update_frequency = None
        self.epsilon = None
        self.decay = None
        self.min_epsilon = None
        self.lr = None
        self.gamma = None
        self.batch_size = None
        self.hidden_layers = None
        self.game_played = None
        self.memory = None

        # SummaryWriter for Logs
        self.summary_writer = None

        # Torch instances
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # Set parameters and create torch instances
        self.set_parameters(**self._DEFAULT_VALUES)

        # Temporary attributes
        self.last_state = None
        self.last_action = None
        self.rewards = []
        self.losses = []
        self.last_rewards = []
        self.last_losses = []


    def play(self, state):
        """Choose randomly in function of epsilon between a random action
        or the action having the best q_value.
        - state : list of values describing the game
        """
        # Random action
        if random.random() < self.epsilon:
            action = random.randrange(self.num_outputs)
            action = torch.tensor([[action]], device=device, dtype=torch.long)
        
        # Best action 
        else:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)

        self.last_state = state
        self.last_action = action
        return action


    def update(self, next_state, reward, done):
        """Update memory, compute loss, optimize_model and update target_net 
        and epsilon parameters at each update_frequency game played.
        """
        # Convert to tensor
        next_state = torch.tensor(next_state, device=device).view(1, -1)
        reward = torch.tensor([reward], device=device)
        done = torch.tensor([done], device=device)

        # Optimize
        self._add_to_memory(self.last_state, self.last_action, next_state, 
            reward, done)
        loss = self._optimize_model()

        # Logs memory
        self.rewards += [reward.item()]
        self.losses += [loss]

        # Game over
        if done:

            self.game_played += 1

            # Epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

            # Reward & Loss
            game_reward = sum(self.rewards)
            game_loss = sum(self.losses)
            self.rewards = []
            self.losses = []
            self.last_rewards += [game_reward]
            self.last_losses += [game_loss]

            # Logs
            self.summary_writer.add_scalar('Reward', game_reward, 
                self.game_played)
            self.summary_writer.add_scalar('Loss', game_loss, 
                self.game_played)
            self.summary_writer.add_scalar('Epsilon', self.epsilon, 
                self.game_played)

            # Each update_frequency games
            if self.game_played % self.update_frequency == 0:

                # Copy weights 
                self._update_target_net()

                # Display
                print(f'Game played: {self.game_played}, '
                    f'Epsilon: {self.epsilon}, '
                    f'Mean reward: {mean(self.last_rewards)}, '
                    f'Mean loss: {mean(self.last_losses)} ')

                # Reset after display
                self.last_rewards = []
                self.last_losses = []


    def set_parameters(self, **parameters):
        """Set all couple (k, v) in **parameters as self.k = v. 
        If lr is changed, create a new optimizer.
        If hidden_layer is changed, create new neural_networks and optimizer.
        If name is changed, create new summary_writer.
        - Any k=v couple values, where k is the name of the parameter, and
        v the value.
        """
        flag_optimizer = False
        flag_neural_networks = False
        flag_summary_writer = False

        for param, value in parameters.items():
            
            # Param valid handle
            if param not in self._DEFAULT_VALUES.keys():     
                raise Exception(f'Parameter {param} not known.')
            
            # Flag handle
            if getattr(self, param) != value:
                if param == 'lr':
                    # If lr modify, we have to recreate optimizer
                    flag_optimizer = True

                elif param == 'hidden_layers':
                    # If layers modify, we have to recreate neural networks
                    # and optimizer
                    flag_neural_networks = True
                    flag_optimizer = True 

                elif param == 'name':
                    # if name is modified, path to the summary_writer change
                    flag_summary_writer = True

            # Set value
            setattr(self, param, value)

        # Create torch instances
        if flag_neural_networks:
            self._create_neural_networks()
            # Print model
            summary(self.policy_net, (1, self.num_inputs))

        if flag_optimizer:
            self._create_optimizer()

        # Create SummaryWriter
        if flag_summary_writer:
            self._reset_summary_writer()



    def get_parameters(self):
        """Return all parameters values.
        """
        params = {}
        for p in self._DEFAULT_VALUES.keys():
            params[p] = getattr(self, p)
        return params


    def save(self, directory='saves/'):
        """Save the model, optimizer and the trainer parameters.
        """
        # Create dirpath for temporary dir
        if directory[-1] != '/':
            directory += '/'
        dirpath = directory + self.name + '/'

        if not os.path.exists(dirpath): 
            os.makedirs(dirpath)
        else:
            raise Exception(f'Path {dirpath} already exists.')

        # DQNs & Optimizer
        torch.save(self.policy_net.state_dict(), f'{dirpath}dqn.pth')
        torch.save(self.optimizer.state_dict(), f'{dirpath}optimizer.pth')

        # Trainer pamameters
        params = {}
        for p in self._DEFAULT_VALUES.keys():
            params[p] = getattr(self, p)

        with open(f'{dirpath}parameters.pick', 'wb') as file:
            pickle.dump(params, file)

        # Zip the saves in one .zip archive
        zippath = f'{directory}{self.name}'
        shutil.make_archive(zippath, 'zip', dirpath)

        # Remove the directory dirpath and files inside
        shutil.rmtree(dirpath)

        # Display
        print(f'Model saved at {zippath}.zip')


    def load(self, filename, directory_tmp='saves/tmp/'):
        """Load the model, optimizer and trainer parameters.
        """
        # Verify path
        if not os.path.exists(filename):
            raise IOError(f'Filename {filename} does not exists.')

        if os.path.exists(directory_tmp):
            raise Exception(
                f'Path {directory_tmp} already exists, \
                please choose a non-existant path.')

        if directory_tmp[-1] != '/':
            directory_tmp += '/'

        # Unzip the archive
        shutil.unpack_archive(filename, directory_tmp, 'zip')
        
        # Trainer parameters
        with open(f'{directory_tmp}parameters.pick', 'rb') as file:
            params = pickle.load(file)
        self.set_parameters(**params)

        # DQNs & Optimizer
        self.policy_net.load_state_dict(
            torch.load(f'{directory_tmp}dqn.pth'))
        self.policy_net.eval()

        self._update_target_net()
        self.target_net.eval()

        self.optimizer.load_state_dict(
            torch.load(f'{directory_tmp}optimizer.pth'))


        # Remove the directory directory_tmp and files inside
        shutil.rmtree(directory_tmp)

        # Display
        print(f'Model {self.name} loaded from {filename}.')


    def _optimize_model(self):
        """Train the model by selecting a random subset of combinaison
        (state, action) in his memory, calcul the loss from q_values,
        and apply back-propagation.
        """
        # Check that there is enough plays in self.experiences
        if len(self.memory) < self.batch_size:
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
        with torch.no_grad():
            next_q_values[~done_batch] = self.target_net(
                next_state_batch[~done_batch]).max(1)[0].detach()

        # Compute expected Q-value
        expected_q_values = (next_q_values * self.gamma) + reward_batch

        # Compute loss
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        if len(self.memory) >= self.size_max_memory:
            self.memory.pop(0)
        self.memory += [Transition(state, action, next_state, reward, done)]


    def _reset_summary_writer(self):
        """Create a new folder at ./logs/name/. If no data were written in
        the old folder, delete it.
        """
        dirname = f'./logs/{self.name}/'
        filename = os.listir(dirname)[0]

        # If no data in old dirname, delete directory
        if os.stat(filename).st_size == 40:
            shutil.rmtree(dirname)

        self.summary_writer = SummaryWriter(log_dir=f'./logs/{self.name}/')


    def _update_target_net(self):
        """Copy the weights and biais from policy_net to target_net
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def _create_neural_networks(self):
        """Create policy_net and target_net, and copy weights from policy_net
        to target_net.
        """
        self.policy_net = DeepQNetwork(self.num_inputs,
                                        self.hidden_layers, 
                                        self.num_outputs).to(device)
        self.target_net = DeepQNetwork(self.num_inputs,
                                        self.hidden_layers, 
                                        self.num_outputs).to(device)
        self._update_target_net()


    def _create_optimizer(self):
        """Create optimizer with lr. Neural network has to be create before
        calling this method.
        """        
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
            lr=self.lr, eps=1e-7)
