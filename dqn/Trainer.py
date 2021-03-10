import os
import pickle
import shutil
from time import sleep
import numpy as np
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

from dqn.DeepQNetwork import DeepQNetwork



class Trainer():

    TIME_TO_SLEEP = 0
    NB_LAST_GAMES = 20
    TRAINING_PARAMS = ['epsilon', 'decay', 'min_epsilon']
    HYPERPARAMETERS = ['lr', 'gamma', 'batch_size', 'min_experiences',
                        'max_experiences', 'hidden_units']

    def __init__(self, qapp, env):  
        """Initalize the trainer and create the DeepQNetworks
        - qapp : Main QWidget window
        - env : gym env
        """
        self.qapp = qapp
        self.env = env

        self.run_name = ''
        self.epsilon = 0
        self.decay = 0
        self.min_epsilon = 0
        self.nb_iter_prev = 0
        self.TrainNet = None
        self.TargetNet = None

    def run_n_games(self, N):
        """Play N games and train the DeepQNetwork
        - N : Number of games to play
        """
        # Summary & Log Writer init
        self.TrainNet.model.summary()
        log_writer = SummaryWriter(log_dir=f'./logs/{self.run_name}/')

        # Make N games
        total_rewards = np.empty(N)
        mean_losses = np.empty(N)

        for n in range(N):
            step = self.nb_iter_prev + n

            # Play one game and update self.epsilon, rewards & losses
            total_reward, mean_loss = self.play_one_game()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

            total_rewards[n] = total_reward
            avg_rewards = total_rewards[max(0, n - self.NB_LAST_GAMES):(n + 1)].mean()

            mean_losses[n] = mean_loss
            avg_losses = mean_losses[max(0, n - self.NB_LAST_GAMES):(n + 1)].mean()

            # Make the summary
            log_writer.add_scalar('Reward', total_reward, step)
            log_writer.add_scalar(f'Avg Rewards (last {self.NB_LAST_GAMES})', avg_rewards, step)
            log_writer.add_scalar('Loss', mean_loss, step)

            # Each self.NB_LAST_GAMES games, display information and update the model
            if (step + 1) % self.NB_LAST_GAMES == 0:
                print(f'episode: {step + 1}, \
                        eps: {self.epsilon}, \
                        avg reward (last {self.NB_LAST_GAMES}): {avg_rewards}, \
                        avg losses (last {self.NB_LAST_GAMES}): {avg_losses}')
                self.TargetNet.copy_weights(self.TrainNet)

        # End of the training
        self.nb_iter_prev += N
        self.env.close()
        self._save_in_zip()


    def play_one_game(self):
        """Play a game with DeepQNetwork agent and train it.
        """
        # Initialize the game
        rewards = 0
        done = False
        observations = self.env.reset()
        losses = list()

        while not done:
            # Choose action in function of observation and play it
            action = self.TrainNet.get_action(observations, self.epsilon)
            prev_observations = observations
            observations, reward, done, _ = self.env.step(action)
            rewards += reward

            self.qapp.processEvents()
            sleep(self.TIME_TO_SLEEP)

            # Reset the game if the gym environnement is finished
            if done:
                self.env.reset()

            # Add this experience to the list of last experiences
            exp = {'s': prev_observations, 'a': action,
                'r': reward, 's2': observations, 'done': done}
            self.TrainNet.add_experience(exp)

            # Train the model and retireve the loss
            loss = self.TrainNet.train(self.TargetNet)
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.numpy())

        return rewards, np.mean(losses)


    def set_params(self, **params):
        """This function modify parameters in the Trainer and DQNs.
        To modify a parameter, call the function with args like :
        param1=value1, param2=value2, ...
        Where param could be :
        - ['epsilon', 'decay', 'min_epsilon', 'nb_iter_prev'] for Trainer.
        - ['lr', 'gamma', 'batch_size', 'min_experiences', 'max_experiences',
        'hidden_units'] for DQNs
        """
        for k, v in params.items():

            if k == 'run_name':
                self.run_name = v

            elif k in self.TRAINING_PARAMS:
                if getattr(self, k) != v:
                    setattr(self, k, v)

            elif k in self.HYPERPARAMETERS:
                if getattr(self.TrainNet, k) != v:
                    
                    setattr(self.TrainNet, k, v)
                    setattr(self.TargetNet, k, v)

                    if k == 'hidden_units':
                        self.TrainNet.create_neural_net()
                        self.TargetNet.create_neural_net()

            else:
                raise Exception(f'Parameter {k} not known.')

    def get_params(self):
        params = {'run_name': self.run_name}
        params.update({p: getattr(self, p) for p in self.TRAINING_PARAMS})
        params.update({p: getattr(self.TrainNet, p) for p in self.HYPERPARAMETERS})
        return params


    def new_session(self, run_name, epsilon, decay, min_epsilon, lr, gamma,
            batch_size, min_experiences, max_experiences, hidden_units):
        """Create a new session with those parameters
        - run_name** : name of the training, if None, take the date
        - loading_file : path to a save, if not None, load the params too
        - epsilon* : initial percentage of random plays
        - decay* : decay of epsilon at each game
        - min_epsilon* : minimum value of epsilon
        - lr : learning rate
        - gamma : coefficiant of next_q_value
        - batch_size : size of a batch
        - min_experience : min size of the memory
        - max_experience : max size of the memory
        - hidden_units : size of neural network's hidden layers 
        *Those parameters will be changed to the saved values if loading_file
        is not None.
        **If you want to keep the name of a loading model, you should give
        the same run_name. The run_name isn't loaded from the save.
        """
        # training parameters
        self.run_name = run_name
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

        # Retieve number of state and action values from the gym env
        num_states = len(self.env.observation_space.sample())
        num_actions = self.env.action_space.n        

        # Instanciate the DQNs
        self.TrainNet = DeepQNetwork(run_name, num_states, num_actions,
            lr, gamma, batch_size, min_experiences,
            max_experiences, hidden_units)
        self.TargetNet = DeepQNetwork(run_name + '_target', num_states, 
            num_actions, lr, gamma, batch_size, min_experiences,
            max_experiences, hidden_units)


    def load_session(self, loading_file):
        # Load TrainNet and training_params
        self._load_from_zip(loading_file)

        # load DQN hyperpameters
        lr = self.TrainNet.lr
        gamma = self.TrainNet.gamma
        batch_size = self.TrainNet.batch_size
        min_experiences = self.TrainNet.min_experiences
        max_experiences = self.TrainNet.max_experiences
        hidden_units = self.TrainNet.hidden_units
        
        # Retieve number of state and action values from the gym env
        num_states = len(self.env.observation_space.sample())
        num_actions = self.env.action_space.n
        
        if num_states != self.TrainNet.num_states:
            raise Exception(f'Loaded model has not the same number of states \
                than the env : {num_states} != {self.TrainNet.num_states}.')

        if num_actions != self.TrainNet.num_actions:
            raise Exception(f'Loaded model has not the same number of actions \
                than the env : {num_actions} != {self.TrainNet.num_actions}.')

        # Create TargetNet
        self.TargetNet = DeepQNetwork(self.run_name + '_target', num_states, 
            num_actions, lr, gamma, batch_size, min_experiences,
            max_experiences, hidden_units)
        self.TrainNet.copy_weights(self.TargetNet)



    def _save_in_zip(self, directory='saves/'):
        """Saves the model into directory/name.zip. The archive contains four
        files : 
        -> the keras model into the keras.sm dir
        -> the optimizer weights into the opt_weights.npy
        -> the params of the training into the training_params.pick 
        -> the dqn instance into the dqn.pick
        * If no name provided, use the model name. 
        ** If not directory provided, use the saves/ directory in the root of the project.
        """

        # Create dirpath for temporary dir
        if directory[-1] != '/':
            directory += '/'
        dirpath = directory + self.run_name + '/'

        if not os.path.exists(dirpath): 
            os.makedirs(dirpath)
        else:
            raise Exception(f'Path {dirpath} already exists.')

        # Keras model
        self.TrainNet.model.save(f'{dirpath}keras.sm')
        self.TrainNet.model = None

        # Optimizer weights
        np.save(f'{dirpath}opt_weights.npy', self.TrainNet.optimizer.get_weights())
        self.TrainNet.optimizer = None

        # DQN instance
        with open(f'{dirpath}dqn.pick', 'wb') as file:
            pickle.dump(self.TrainNet, file)

        # Params of the training
        training_params = {'run_name': self.run_name,
                        'epsilon': self.epsilon,
                        'decay': self.decay,
                        'min_epsilon': self.min_epsilon,
                        'nb_iter_prev': self.nb_iter_prev}
        with open(f'{dirpath}training_params.pick', 'wb') as file:
            pickle.dump(training_params, file)

        # Zip the saves in one .zip archive
        zippath = f'{directory}{self.run_name}'
        shutil.make_archive(zippath, 'zip', dirpath)

        # Remove the directory dirpath and files inside
        shutil.rmtree(dirpath)

        # Display
        print(f'Model saved at {zippath}.zip')


    def _load_from_zip(self, filename, directory_tmp='saves/tmp/'):
        """Loads the model from a .zip file containing :
        -> the keras model
        -> the optimizer weights
        -> the params of the training
        -> the dqn instance
        """

        # Verify path
        if not os.path.exists(filename):
            raise IOError(f'Filename {filename} does not exists.')

        if os.path.exists(directory_tmp):
            raise Exception(
                f'Path {directory_tmp} already exists, please choose a non-existant path.')

        if directory_tmp[-1] != '/':
            directory_tmp += '/'

        # Unzip the archive
        shutil.unpack_archive(filename, directory_tmp, 'zip')

        # DQN instance
        self.TrainNet = None
        with open(f'{directory_tmp}dqn.pick', 'rb') as file:
            self.TrainNet = pickle.load(file)

        # Keras model
        self.TrainNet.model = tf.keras.models.load_model(f'{directory_tmp}keras.sm')

        # Optimizer weights
        self.TrainNet.optimizer = tf.optimizers.Adam(self.TrainNet.lr)
        variables = self.TrainNet.model.trainable_variables
        self._load_optimizer_state(
            self.TrainNet.optimizer, f'{directory_tmp}opt_weights.npy', variables)

        # Params of the training
        training_params = None
        with open(f'{directory_tmp}training_params.pick', 'rb') as file:
            training_params = pickle.load(file)

        self.run_name = training_params['run_name']
        self.epsilon = training_params['epsilon']
        self.decay = training_params['decay']
        self.min_epsilon = training_params['min_epsilon']
        self.nb_iter_prev = training_params['nb_iter_prev']

        # Remove the directory directory_tmp and files inside
        shutil.rmtree(directory_tmp)

        # Display
        print(f'Model {self.TrainNet.name} loaded from {filename}.')


    def _load_optimizer_state(self, optimizer, path, model_train_vars):
        '''
        Loads keras.optimizers object state.

        Arguments:
        optimizer --- Optimizer object to be loaded.
        path --- path to the save
        model_train_vars --- List of model variables (obtained using Model.trainable_variables)

        '''

        # Load optimizer weights
        opt_weights = np.load(path, allow_pickle=True)

        # dummy zero gradients
        zero_grads = [tf.zeros_like(w) for w in model_train_vars]
        # save current state of variables
        saved_vars = [tf.identity(w) for w in model_train_vars]

        # Apply gradients which don't do nothing with Adam
        optimizer.apply_gradients(zip(zero_grads, model_train_vars))

        # Reload variables
        [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]

        # Set the weights of the optimizer
        optimizer.set_weights(opt_weights)
