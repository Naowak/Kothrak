import os
import pickle
import shutil
import random
from time import sleep
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


class Trainer():

    PARAMETERS = ['name', 'nb_games', 'update_frequency', 'time_to_sleep',
                    'epsilon', 'decay', 'min_epsilon', 'lr', 'gamma', 
                    'batch_size', 'hidden_layers']
    DEFAULT_VALUES = {'name': datetime.now().strftime("%m%d%y-%H%M"), 
                        'nb_games': 1000, 
                        'update_frequency': 20,
                        'time_to_sleep': 0,
                        'epsilon': 0.99, 
                        'decay': 0.8,
                        'min_epsilon': 0.01, 
                        'lr': 1e-3, 
                        'gamma': 0.99,
                        'batch_size': 32,
                        'hidden_layers': [150],
                        'nb_iter_prev': 0, 
                        'memory': []}

    def __init__(self, env):
        """Initialize the Trainer.
        - env : KothrakEnv instance
        """        
        # Definitive attributes
        self.env = env
        self.size_max_memory = 2000
        self.num_inputs = env.num_observations
        self.num_actions = env.num_actions

        # Initialize attributes (may change if model loaded)
        self.name = None
        self.nb_games = None
        self.update_frequency = None
        self.time_to_sleep = None
        self.epsilon = None
        self.decay = None
        self.min_epsilon = None
        self.lr = None
        self.gamma = None
        self.batch_size = None
        self.hidden_layers = None
        self.nb_iter_prev = None
        self.memory = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.set_parameters(**self.DEFAULT_VALUES) 


    def run(self):
        """Play nb_games and optimize model.
        """
        def update_logs(summary_writer, episode, reward, loss, epsilon):
            summary_writer.add_scalar('Reward', reward, episode)
            summary_writer.add_scalar('Loss', loss, episode)
            summary_writer.add_scalar('Epsilon', epsilon, episode)
        
        # Print model and init summary_writer
        summary(self.policy_net, (1, self.num_inputs))
        summary_writer = SummaryWriter(log_dir=f'./logs/{self.name}/')

        sum_reward = 0

        # Run nb_games
        for n in range(self.nb_games):

            reward, loss = self._run_one_game()

            # Update values and logs
            episode = self.nb_iter_prev + n
            sum_reward += reward
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            update_logs(summary_writer, episode, reward, loss, self.epsilon)
            
            # Each update_frequency print and update target_net
            if (episode + 1) % self.update_frequency == 0:
                print(f'Episode: {episode + 1}, Epsilon: {self.epsilon}, '
                    f'Reward: {reward}, Loss: {loss}, '
                    f'Mean reward: {sum_reward/self.update_frequency}.')
                sum_reward = 0
                self._update_target_net()

        # End of the training
        self.nb_iter_prev += self.nb_games
        self.save()


    def set_parameters(self, create_models=True, **parameters):
        """Set all couple (k, v) in **parameters as self.k = v. Create 
        optimizer and dqns instances if lr or hidden_layer is part in 
        parameters, and have different values than the one existing.
        - Any k=v couple values, where k is a string a v a value.
        """
        flag_nn_opti = False

        # Set attributes
        for param, value in parameters.items():
            if param in self.DEFAULT_VALUES.keys():
                if getattr(self, param) != value:
                    # We change param value
                    setattr(self, param, value)
                    if param in ['hidden_layers', 'lr']:
                        flag_nn_opti = True

            else:
                raise Exception(f'Parameter {param} not known.')

        # Create torch instances
        if create_models and flag_nn_opti:
            self._create_networks_and_optimizer()


    def get_parameters(self):
        """Return all parameters values.
        """
        params = {}
        for p in self.DEFAULT_VALUES.keys():
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
        for p in self.DEFAULT_VALUES.keys():
            params[p] = getattr(self, p)

        with open(f'{dirpath}trainer_parameters.pick', 'wb') as file:
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
        with open(f'{directory_tmp}trainer_parameters.pick', 'rb') as file:
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


    def _run_one_game(self):
        """Play one game and optimize model.
        """
        sum_reward = 0
        done = False
        state = torch.tensor(self.env.reset(), device=device).view(1, -1)
        losses = list()

        # turn_memory = {}

        while not done:

            # Choose action in function of observation and play it
            action = self._select_action(state)
            next_state, reward, done, _ = self.env.step(action.item())

            sum_reward += reward
            next_state = torch.tensor(next_state, device=device).view(1, -1)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            
            # Add transition to memory
            self._add_to_memory(state, action, next_state, reward, done)

            # Compute loss
            loss = self._optimize_model()
            losses += [loss]
            
            # Prepare next state
            state = next_state

            # Wait time_to_sleep second so the user can view the state
            sleep(self.time_to_sleep)
            

        return sum_reward, mean(losses)


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


    def _select_action(self, state):
        """Choose randomly in function of epsilon between a random action
        or the action having the best q_value.
        - state : list of values describing the game
        """
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
            return torch.tensor([[action]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)


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


    def _update_target_net(self):
        """Copy the weights and biais from policy_net to target_net
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def _create_networks_and_optimizer(self):
        """Create policy_net and target_net, and copy weights from policy_net
        to target_net, then create optimizer with lr.
        """
        self.policy_net = DeepQNetwork(self.num_inputs,
                                        self.hidden_layers, 
                                        self.num_actions).to(device)
        self.target_net = DeepQNetwork(self.num_inputs,
                                        self.hidden_layers, 
                                        self.num_actions).to(device)
        self._update_target_net()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
            lr=self.lr, eps=1e-7)



def launch_test():
    """Create an instance of trainer and launch the training to test the class
    """
    import sys
    from kothrak.envs.KothrakEnv import KothrakEnv
    from kothrak.envs.game.MyApp import style
    from PyQt5.QtWidgets import QApplication, QWidget

    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    window = QWidget()
    window.setWindowTitle('Kothrak training')

    env = KothrakEnv(qapp, window)
    window.show()

    trainer = Trainer(env)
    # trainer.load('saves/031421-1523.zip')
    trainer.run()

    qapp.exec_()
