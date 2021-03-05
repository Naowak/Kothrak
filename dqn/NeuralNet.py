import tensorflow as tf

class NeuralNet(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(NeuralNet, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))

        self.hidden_layers = []
        for nb_neurones in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                nb_neurones, activation='relu', kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output