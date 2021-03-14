import os
from collections import namedtuple
import random
from time import sleep
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

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


class Trainer():

    TIME_TO_SLEEP = 0
    NB_GAMES_UPDATE = 20

    def __init__(self, env, name, nb_games, epsilon, decay, min_epsilon,
            num_inputs, num_actions, lr, gamma,
            batch_size, size_min_memory, size_max_memory, hidden_units):
        self.env = env
        self.name = name
        self.nb_games = nb_games
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.nb_iter_prev = 0
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.size_min_memory = size_min_memory
        self.size_max_memory = size_max_memory
        self.hidden_units = hidden_units

        self.policy_net = DeepQNetwork(num_inputs, num_actions).to(device)
        self.target_net = DeepQNetwork(num_inputs, num_actions).to(device)
        self.update_target_net()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []


    def run(self):
        """Play self.nb_games and optimize model."""

        def update_logs(summary_writer, episode, reward, loss, epsilon):
            summary_writer.add_scalar('Reward', reward, episode)
            summary_writer.add_scalar('Loss', loss, episode)
            summary_writer.add_scalar('Epsilon', epsilon, episode)
        
        # Print model and init summary_writer
        summary(self.policy_net, (1, self.num_inputs))
        summary_writer = SummaryWriter(log_dir=f'./logs/{self.name}/')

        # Run nb_games
        for n in range(self.nb_games):

            reward, loss = self.run_one_game()

            # Update values and logs
            episode = self.nb_iter_prev + n
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            update_logs(summary_writer, episode, reward, loss, self.epsilon)
            
            # Each NB_GAMES_UPDATE print and update target_net
            if (episode + 1) % self.NB_GAMES_UPDATE == 0:
                print(f'Episode: {episode + 1}, Epsilon: {self.epsilon}')
                self.update_target_net()

        # End of the training
        self.nb_iter_prev += self.nb_games
        self._save_in_zip()


    def run_one_game(self):
        """Play one game and optimize model."""
        sum_reward = 0
        done = False
        # state = torch.from_numpy(self.env.reset()).double().to(device)
        state = torch.tensor(self.env.reset(), device=device).view(1, -1)
        losses = list()

        while not done:

            # Choose action in function of observation and play it
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action.item())

            sum_reward += reward
            next_state = torch.tensor(next_state, device=device).view(1, -1)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            
            # Add transition to memory
            self.add_to_memory(state, action, next_state, reward, done)

            # Compute loss
            loss = self.optimize_model()
            losses += [loss]
            
            # Prepare next state
            state = next_state

            sleep(self.TIME_TO_SLEEP)
            

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


    def update_target_net(self):
        """Copy the weights and biais from policy_net to target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    # def _save_in_zip(self, directory='saves/'):

    #     # Create dirpath for temporary dir
    #     if directory[-1] != '/':
    #         directory += '/'
    #     dirpath = directory + self.name + '/'

    #     if not os.path.exists(dirpath): 
    #         os.makedirs(dirpath)
    #     else:
    #         raise Exception(f'Path {dirpath} already exists.')

    #     # DQNs
    #     torch.save(self.policy_net.state_dict(), f'{dirpath}dqn.pth')

    #     # Optimizer
    #     torch.save(self.optimizer.state_dict(), f'{dirpath}optimizer.pth')



    #     # Keras model
    #     self.TrainNet.model.save(f'{dirpath}keras.sm')
    #     model_tmp = self.TrainNet.model
    #     self.TrainNet.model = None

    #     # Optimizer weights
    #     np.save(f'{dirpath}opt_weights.npy', self.TrainNet.optimizer.get_weights())
    #     optimizer_tmp = self.TrainNet.optimizer
    #     self.TrainNet.optimizer = None

    #     # DQN instance
    #     with open(f'{dirpath}dqn.pick', 'wb') as file:
    #         pickle.dump(self.TrainNet, file)

    #     # Params of the training
    #     training_params = {'run_name': self.run_name,
    #                     'nb_games': self.nb_games,
    #                     'epsilon': self.epsilon,
    #                     'decay': self.decay,
    #                     'min_epsilon': self.min_epsilon,
    #                     'nb_iter_prev': self.nb_iter_prev}
    #     with open(f'{dirpath}training_params.pick', 'wb') as file:
    #         pickle.dump(training_params, file)

    #     # Current trainer retrieve model and optimizer for next training
    #     self.TrainNet.model = model_tmp
    #     self.TrainNet.optimizer = optimizer_tmp

    #     # Zip the saves in one .zip archive
    #     zippath = f'{directory}{self.run_name}'
    #     shutil.make_archive(zippath, 'zip', dirpath)

    #     # Remove the directory dirpath and files inside
    #     shutil.rmtree(dirpath)

    #     # Display
    #     print(f'Model saved at {zippath}.zip')






def launch():
    import sys
    from kothrak.envs.KothrakEnv import KothrakEnv
    from kothrak.envs.game.MyApp import style
    from PyQt5.QtWidgets import QApplication, QWidget

    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    window = QWidget()
    window.setWindowTitle('Kothrak training')
    window.setObjectName('trainer_bg')

    env = KothrakEnv(qapp, window)

    trainer = Trainer(env, 'name', 100, 0.99, 0.8, 0.05, 40, 6, 0.001, 0.99, 32, 100, 1000, [])
    print('hello')
    trainer.run()

    # qapp.exec_()
