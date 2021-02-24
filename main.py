import sys
import numpy as np
import datetime
import gym
import tensorflow as tf

from kothrak.envs.game import MyApp
from kothrak.envs.KothrakEnv import KothrakEnv
from dqn.DeepQNetwork import DeepQNetwork


def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    """Play a game with DeepQNetwork agent and train it.
        - env : Environnement gym
        - TrainNet : DeepQNetwork original
        - TargetNet : Copy of DeepQNetwork
        - epsilon : probability of a random action
        - copy_step : number of iteration before training
    """
    # Initialize the game
    rewards = 0
    iteration = 0
    done = False
    observations = env.reset()
    losses = list()

    while not done:
        # Choose action in function of observation and play it
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        
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

        # If we made enough iteration, update the model
        iteration += 1
        if iteration % copy_step == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards, np.mean(losses)


def main():
    env = gym.make('kothrak-v0')

    # tuning hyperparameters
    lr = 1e-2
    gamma = 0.90
    copy_step = 5
    batch_size = 32
    min_experiences = 100
    max_experiences = 10000
    hidden_units = [200, 200]
    epsilon = 0.95
    decay = 0.95
    min_epsilon = 0.05
    
    # Retieve number of state and action values
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n

    # Prepare logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/DeepQNetwork/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Create DQNs
    TrainNet = DeepQNetwork(num_states, num_actions, hidden_units, gamma,
                   max_experiences, min_experiences, batch_size, lr)
    TargetNet = DeepQNetwork(num_states, num_actions, hidden_units, gamma,
                    max_experiences, min_experiences, batch_size, lr)

    # Make N games
    N = 50000
    total_rewards = np.empty(N)

    for n in range(N):
        # Play one game and update epsilon & rewards
        total_reward, losses = play_game(
            env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        epsilon = max(min_epsilon, epsilon * decay)
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        
        # Make the summary
        summary_writer.as_default()
        tf.summary.scalar('episode reward', total_reward, step=n)
        tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
        tf.summary.scalar('average loss)', losses, step=n)
        summary_writer.close()

        # Display infos each 100 games
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    
    # End of the training
    print("avg reward for last 100 episodes:", avg_rewards)
    # make_video(env, TrainNet)
    env.close()



if __name__ == '__main__':

    # values = ['ENV', 'GAME']
    RUN = 'GAME'

    if RUN == 'GAME':
        MyApp.run()

    elif RUN == 'ENV':
        main()
