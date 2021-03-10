import tensorflow as tf
import numpy as np

from dqn.NeuralNet import NeuralNet



class DeepQNetwork:

    def __init__(self, name, num_states, num_actions, lr, gamma, batch_size, 
            min_experiences, max_experiences, hidden_units):
        self.name = name
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_experiences = min_experiences
        self.max_experiences = max_experiences
        self.hidden_units = hidden_units

        self.optimizer = tf.optimizers.Adam(lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.model = None
        self.create_neural_net()

    def create_neural_net(self):
        self.model = NeuralNet(self.num_states, self.hidden_units, self.num_actions)
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
