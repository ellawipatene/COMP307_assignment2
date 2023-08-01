import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, hidden_layer_biases, output_layer_biases, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.hidden_layer_biases = hidden_layer_biases
        self.output_layer_biases = output_layer_biases

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1 / (1 + np.exp(-input))
        return output

    # Calculate the derivative of the sigmoid func
    def derivative(self, x):
        output = x * (1 - x)
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            weighted_sum = 0.0
            for j in range(len(self.hidden_layer_weights)):
                weighted_sum += self.hidden_layer_weights[j,i] * inputs[j]
            output = self.sigmoid(weighted_sum + self.hidden_layer_biases[i])
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            weighted_sum = 0.0
            for j in range(len(self.output_layer_weights)):
                weighted_sum += self.output_layer_weights[j,i] * hidden_layer_outputs[j]
            output = self.sigmoid(weighted_sum + self.output_layer_biases[i])
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        output_layer_betas = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            output_layer_betas[i] = desired_outputs[i] - output_layer_outputs[i]
        #print('OL betas: ', output_layer_betas)

        hidden_layer_betas = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            temp_error = 0.0
            for j in range(self.num_outputs):
                temp_error += self.output_layer_weights[i][j] * output_layer_betas[j]
            hidden_layer_betas[i] = temp_error
        #print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                delta_output_layer_weights[i][j] = self.learning_rate * output_layer_betas[j] * self.derivative(output_layer_outputs[j]) * hidden_layer_outputs[i]


        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                delta_hidden_layer_weights[i][j] = self.learning_rate * hidden_layer_betas[j] * self.derivative(hidden_layer_outputs[j]) * inputs[i]

        # This is for working out the changes in biases
        delta_output_layer_biases = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            delta_output_layer_biases[i] = self.learning_rate * output_layer_betas[i] * self.derivative(output_layer_outputs[i])

        delta_hidden_layer_biases = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            delta_hidden_layer_biases[i] = self.learning_rate * hidden_layer_betas[i] * self.derivative(hidden_layer_outputs[i])

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_biases, delta_hidden_layer_biases

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_biases, delta_hidden_layer_biases):
        # Update the weights:
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                self.output_layer_weights[i][j] += delta_output_layer_weights[i][j]

        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                self.hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j]

        # Update the values of the biases:
        for i in range(self.num_outputs):
            self.output_layer_biases[i] += delta_output_layer_biases[i]
        
        for i in range(self.num_hidden):
            self.hidden_layer_biases[i] += delta_hidden_layer_biases[i]

    
    def train(self, instances, desired_outputs, epochs):
        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_biases, delta_hidden_layer_biases = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])

                predicted_class = 0
                if output_layer_outputs[0] > output_layer_outputs[1] and output_layer_outputs[0] > output_layer_outputs[2]:
                    predicted_class = 0
                elif output_layer_outputs[1] > output_layer_outputs[0] and output_layer_outputs[1] > output_layer_outputs[2]:
                    predicted_class = 1
                else:
                    predicted_class = 2
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_biases, delta_hidden_layer_biases)

            # Print new weights
            #print('Hidden layer weights \n', self.hidden_layer_weights)
            #print('Output layer weights  \n', self.output_layer_weights)

            #print('Hidden layer biases \n', self.hidden_layer_biases)
            #print('Output layer biases  \n', self.output_layer_biases)

            # TODO: Print accuracy achieved over this epoch
            counter = 0
            for i in range(len(predictions)):
                if desired_outputs[i][predictions[i]] == 1.0:
                    counter += 1
            acc = counter / len(instances)
            print('acc = ', acc)

    def train_one(self, instance, desired_output):
        hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
        delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_biases, delta_hidden_layer_biases = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_output)

        # We use online learning, i.e. update the weights after every instance.
        self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights, delta_output_layer_biases, delta_hidden_layer_biases)

        print('Hidden layer weights \n', self.hidden_layer_weights)
        print('Output layer weights  \n', self.output_layer_weights)


    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            predicted_class = None
            if output_layer_outputs[0] > output_layer_outputs[1] and output_layer_outputs[0] > output_layer_outputs[2]:
                predicted_class = 0
            elif output_layer_outputs[1] > output_layer_outputs[0] and output_layer_outputs[1] > output_layer_outputs[2]:
                predicted_class = 1
            else:
                predicted_class = 2
            predictions.append(predicted_class)
        return predictions