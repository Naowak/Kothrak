import sys
import numpy as np
import datetime
import gym
import tensorflow as tf
from time import sleep
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

from kothrak.envs.game.MyApp import MyApp, style, run
from kothrak.envs.game.Utils import APP_PIXDIM, NB_CELLS
from dqn.DeepQNetwork import DeepQNetwork

TIME_TO_SLEEP = 0.1

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
        sleep(TIME_TO_SLEEP)
        
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


def run_n_games(qapp, env, N=3000):

    # tuning hyperparameters
    lr = 1e-2
    gamma = 0.99
    batch_size = 32
    min_experiences = 100
    max_experiences = 10000
    hidden_units = [40, 30, 20]
    epsilon = 0.99
    decay = 0.999
    min_epsilon = 0.05
    
    # Retieve number of state and action values
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n

    # Prepare logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Create DQNs
    TrainNet = DeepQNetwork(num_states, num_actions, hidden_units, gamma,
                   max_experiences, min_experiences, batch_size, lr)
    TargetNet = DeepQNetwork(num_states, num_actions, hidden_units, gamma,
                    max_experiences, min_experiences, batch_size, lr)

    # Make N games
    total_rewards = np.empty(N)
    mean_losses = np.empty(N)
    nb_last_games = 50

    for n in range(N):
        # Play one game and update epsilon & rewards
        total_reward, mean_loss = play_game(qapp, env, TrainNet, TargetNet, epsilon)
        epsilon = max(min_epsilon, epsilon * decay)

        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - nb_last_games):(n + 1)].mean()

        mean_losses[n] = mean_loss
        avg_losses = mean_losses[max(0, n - nb_last_games):(n + 1)].mean()
        
        # Make the summary
        summary_writer.as_default()
        tf.summary.scalar('episode reward', total_reward, step=n)
        tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
        tf.summary.scalar('average loss)', mean_loss, step=n)
        summary_writer.close()

        # Each nb_last_games games, display information and update the model
        if (n+1) % nb_last_games == 0:
            print(f'episode: {n+1}, eps: {epsilon}, avg reward (last {nb_last_games}): {avg_rewards}, \
                avg losses (last {nb_last_games}): {avg_losses}')
            TargetNet.copy_weights(TrainNet)
    
    # End of the training
    print("avg reward for last 100 episodes:", avg_rewards)
    env.close()


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

    # Add button to launch the trainig to the interface
    button = QPushButton('Play N Games', window)
    button.setGeometry(QtCore.QRect(APP_PIXDIM[0], 100, 100, 40))
    button.clicked.connect(lambda : run_n_games(qapp, env))

    # Launch the PyQt programm
    window.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    # values = ['ENV', 'GAME']
    RUN = 'GAME'

    if RUN == 'GAME':
        run()

    elif RUN == 'ENV':
        main()
