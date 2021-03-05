import tensorflow as tf
import numpy as np
import os
import pickle
import shutil

from dqn.NeuralNet import NeuralNet



class DeepQNetwork:
    
    def __init__(self, name, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.name = name
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        self.gamma = gamma
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.lr = lr

        self.optimizer = tf.optimizers.Adam(lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.model = NeuralNet(num_states, hidden_units, num_actions)
        self.model.build((self.batch_size, self.num_states))
    
    def predict(self, inputs):
        """Predict q_value for each action in function of inputs."""
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        """Train the model by selecting a random subset of combinaison
        (state, action) in his last experiences, calcul the loss from q_values,
        and apply back-propagation."""
        # Check that there is enough plays in self.experiences
        if len(self.experience['s']) < self.min_experiences:
            return 0

        # Select self.batch_size random experience
        ids = np.random.randint(low=0, high=len(
            self.experience['s']), size=self.batch_size)

        # Retrieve state, action, reward, next_state and done for each experience
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        # Predict next_q and compute q_value (Bellman equation)
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        # Compute the loss
        with tf.GradientTape() as tape:
            # Use train network to predict score for each action
            # Filter them with one-hot-encoded action realised
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(
                tf.square(actual_values - selected_action_values))

        # Apply gradient descent
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        """Choose randomly between a random action or the action having 
        the best q_value."""
        if np.random.random_sample() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])
    
    def add_experience(self, exp):
        """Add a new experience to the list of last experiences, remove 
        the first ones if there is no more rooms."""
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, dqn):
        """Copy the neural network from the model into dqn."""
        variables1 = self.model.trainable_variables
        variables2 = dqn.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
    




def save_dqn(dqn, params_iter, name=None, directory='saves/'):
    """Saves the model into two files : a .h5 file for the keras model, and a .pick file for the DeepQNetwork instance.
    Compress those two files in a .zip file.
    If no name provided, use the model name. If not directory provided, use the saves/ directory in the root of the project."""
    
    # Retrieve name if not provided
    if name is None:
        name = dqn.name

    # Create dirpath for temporary dir
    if directory[-1] != '/':
        directory += '/'
    dirpath = directory + name + '/'

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        raise Exception(f'Path {dirpath} already exists.')

    # Create .sm, .npy and .pick saves
    dqn.model.save(f'{dirpath}keras.sm')
    dqn.model = None

    np.save(f'{dirpath}opt_weights.npy', dqn.optimizer.get_weights())
    dqn.optimizer = None

    with open(f'{dirpath}dqn.pick', 'wb') as file:
        pickle.dump(dqn, file)
    
    with open(f'{dirpath}params_iter.pick', 'wb') as file:
        pickle.dump(params_iter, file)

    # Zip the saves in one .zip archive
    zippath = f'{directory}{name}'
    shutil.make_archive(zippath, 'zip', dirpath)

    # Remove the directory dirpath and files inside
    shutil.rmtree(dirpath)

    print(f'Model saved at {zippath}.zip')


def load_dqn(filename, directory_tmp='saves/tmp/'):
    """Loads the model from a .zip file containing .h5 and .pick saves."""

    # Verify path
    if not os.path.exists(filename):
        raise IOError(f'Filename {filename} does not exists.')
    
    if os.path.exists(directory_tmp):
        raise Exception(f'Path {directory_tmp} already exists, please choose a non-existant path.')

    if directory_tmp[-1] != '/':
        directory_tmp += '/'
    
    # Unzip the archive
    shutil.unpack_archive(filename, directory_tmp, 'zip')

    # Load the DeepQNetwork instance from the .pick, the keras model from .sm file
    # the optimizer from .npy file and params_iter from .pick
    dqn = None
    with open(f'{directory_tmp}dqn.pick', 'rb') as file:
        dqn = pickle.load(file)

    dqn.model = tf.keras.models.load_model(f'{directory_tmp}keras.sm')
    
    dqn.optimizer = tf.optimizers.Adam(dqn.lr)
    variables = dqn.model.trainable_variables
    _load_optimizer_state(dqn.optimizer, f'{directory_tmp}opt_weights.npy', variables)

    params_iter = None
    with open(f'{directory_tmp}params_iter.pick', 'rb') as file:
        params_iter = pickle.load(file)

    # Remove the directory directory_tmp and files inside
    shutil.rmtree(directory_tmp)

    print(f'Model {dqn.name} loaded from {filename}.')
    return dqn, params_iter


def _load_optimizer_state(optimizer, path, model_train_vars):
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
