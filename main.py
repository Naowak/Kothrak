import sys
import datetime
import pickle
import numpy as np
from time import sleep
import gym
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

from kothrak.envs.game.MyApp import MyApp, style, run
from kothrak.envs.game.Utils import APP_PIXDIM
from dqn.DeepQNetwork import DeepQNetwork, save_dqn, load_dqn

TIME_TO_SLEEP = 0.01
NB_GAMES = 10000
NB_LAST_GAMES = 20

def play_game(qapp, env, TrainNet, TargetNet, epsilon):
    """Play a game with DeepQNetwork agent and train it.
        - env : Environnement gym
        - TrainNet : DeepQNetwork original
        - TargetNet : Copy of DeepQNetwork
        - epsilon : probability of a random action
        - copy_step : number of iteration before training
    """
    # Initialize the game
    rewards = 0
    done = False
    observations = env.reset()
    losses = list()

    while not done:
        # Choose action in function of observation and play it
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward

        qapp.processEvents()
        # sleep(TIME_TO_SLEEP)
        
        # Reset the game if the gym environnement is finished
        if done:
            env.reset()

        # Add this experience to the list of last experiences
        exp = {'s': prev_observations, 'a': action,
               'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)

        # Train the model and retireve the loss
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())

    return rewards, np.mean(losses)


def run_n_games(qapp, env, run_name='', params_iter=None, loading_file=None, N=NB_GAMES):

    # If no run_name given, take the date
    if run_name == '':
        run_name = datetime.datetime.now().strftime("%m%d%Y-%H%M")

    # Create or load DQNs
    if loading_file is None:
        # Create DQNs
        lr = 1e-3
        gamma = 0.99
        batch_size = 32
        min_experiences = 100
        max_experiences = 2000
        hidden_units = [120, 120, 120, 120]
        
        # Retieve number of state and action values from the gym env
        num_states = len(env.observation_space.sample())
        num_actions = env.action_space.n

        # Instanciate the DQNs
        TrainNet = DeepQNetwork(run_name, num_states, num_actions, hidden_units, gamma,
                        max_experiences, min_experiences, batch_size, lr)
        TargetNet = DeepQNetwork(run_name + '_target', num_states, num_actions, hidden_units, gamma,
                        max_experiences, min_experiences, batch_size, lr)

    else:
        # Load TrainNet and params_iter
        TrainNet, params_iter = load_dqn(loading_file)

        # load hyperpameters
        lr = TrainNet.lr
        gamma = TrainNet.gamma
        batch_size = TrainNet.batch_size
        min_experiences = TrainNet.min_experiences
        max_experiences = TrainNet.max_experiences
        hidden_units = TrainNet.hidden_units
        num_states = TrainNet.num_states
        num_actions = TrainNet.num_actions

        # Create TargetNet
        TargetNet = DeepQNetwork(run_name + '_target', num_states, num_actions, hidden_units, gamma,
                        max_experiences, min_experiences, batch_size, lr)
        TrainNet.copy_weights(TargetNet)

    TrainNet.model.summary()

    # Prepare logs writer
    log_writer = SummaryWriter(log_dir=f'./logs/{run_name}/')

    # Make N games
    total_rewards = np.empty(N)
    mean_losses = np.empty(N)
    
    if params_iter is None :
        epsilon = 0.99
        decay = 0.9998
        min_epsilon = 0
        n_begin = 0
    else:
        epsilon = params_iter['epsilon']
        decay = params_iter['decay']
        min_epsilon = params_iter['min_epsilon']
        n_begin = params_iter['n_iter']

    for n in range(N):
        # Play one game and update epsilon & rewards
        total_reward, mean_loss = play_game(qapp, env, TrainNet, TargetNet, epsilon)
        epsilon = max(min_epsilon, epsilon * decay)

        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - NB_LAST_GAMES):(n + 1)].mean()

        mean_losses[n] = mean_loss
        avg_losses = mean_losses[max(0, n - NB_LAST_GAMES):(n + 1)].mean()
        
        # Make the summary
        log_writer.add_scalar('Reward', total_reward, n_begin + n)
        log_writer.add_scalar(f'Avg Rewards (last {NB_LAST_GAMES})', 
                              avg_rewards, n_begin + n)
        log_writer.add_scalar('Loss', mean_loss, n_begin + n)

        # Each NB_LAST_GAMES games, display information and update the model
        if (n+1) % NB_LAST_GAMES == 0:
            print(f'episode: {n_begin + n+1}, \
                    eps: {epsilon}, \
                    avg reward (last {NB_LAST_GAMES}): {avg_rewards}, \
                    avg losses (last {NB_LAST_GAMES}): {avg_losses}')
            TargetNet.copy_weights(TrainNet)
    
    # End of the training
    env.close()
    
    # Save the model
    params_iter = {'epsilon': epsilon, 
                'decay': decay, 
                'min_epsilon': min_epsilon,
                'n_iter': n+1}
    save_dqn(TrainNet, params_iter)

def main():    
    # Create the main window
    qapp = QApplication(sys.argv)
    qapp.setStyleSheet(style)
    window = QWidget()
    window.resize(1000, APP_PIXDIM[1])
    window.setWindowTitle('Kothrak training')

    # Load the environnement
    game = MyApp(window)
    env = gym.make('kothrak-v0')
    env.set_game(game)

    # Get the run name in the args
    run_name = ''
    if len(sys.argv) > 1:
        run_name = sys.argv[1]

    # Add button to launch the trainig to the interface
    button = QPushButton('Play N Games', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0], 100, 100, 40))
    button.clicked.connect(lambda : run_n_games(qapp, env, 
                                                run_name=run_name, 
                                                loading_file='saves/test_continue_training.zip'))

    # Launch the PyQt programm
    window.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    # values = ['ENV', 'GAME']
    RUN = 'ENV'

    if RUN == 'GAME':
        run()

    elif RUN == 'ENV':
        main()
