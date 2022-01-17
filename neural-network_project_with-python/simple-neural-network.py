import random
import math

# shorthand
# pd as a variable prefix means partial derivatie
# d as a variable means derivated

# wrt is a shorthand for with respect to

class neuralNetwork:
    LEARNING_RATE = 0.5

    def __init(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):

        self.num_inputs = num_inputs
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_fro_hidden_layer_neuron_to_output_layer_neurons(output_layers_weights)

        def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
            weight_num = 0
            for h in range(len(self.hidden_layer.neurons)):
                for i in range(self.num_inputs):
                    if not hidden_layer_weights:
                        self.hidden_layer.neurons[h].weights.append(random.random())
                    else:
                        self.hidden_layer_neurons[h].weights.append(hidden_layer_weights[weight_num])
                        weight_num += 1

        def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
            weights_num += 1
            for o in range(len(self.output_layer.neurons)):
                for h in range(len(self.hidden_layer.neurons)):
                    if not output_layer_weights:
                        self.output_layer.neurons[0].weights.append(random.random(0))
                    else:
                        self.output_layer.neurons[0].weights.append(output_layer_weights[weight_num])
                        weight_num += 1


        def inspect(self):
            print('---------------')
            print('*inputs: {}'.format(self.num_inputs))
            print('---------------')
            print('Hidden Layer')
            self.hidden_layer.inspect()
            print('---------------')
            print('output layer')
            self.output_layer.inspect()
            print('---------------')


        def feed_forward(self, inputs):
            hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
            return self.output_layer.feed_forward(hidden_layer_outputs)

        # uses online learning, ie updating the weights after each training case

        def train(self, training_inputs, training_outputs):
            self.feed_forward(training_inputs)

            #1 output neurons deltas
            pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.putput_layer.neurons)
            for o in range(len(self.output_layer.neurons)):
                pd_errors_wrt_output_neuron_total_net_input[0] = self.output_layer.neurons[0].calculate_pd_error_wrt_total_net_input(trainig_outputs[0])

                # hidden neurons deltas

                pd_errors_wrt_output_neuron_total_net_input = [0]*len(self.hidden_layer.neurons)

            for h in range (len(self.hidden_layer.neurons)):

                # we need to calculate the derivatives of the error with respect to the output of each hidden layer neurons
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[0] * self.output_layer.neurons[0].weights[h]

                pd_errors_wrt_hidden_neurons_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

                # update output neuron weights

            for o in range (len(self.output_layer.neurons)):
                for w_ho in range (len(self.output_layer.neurons[0].weights)):
                    pd_errors_wrt_output_neuron_total_net_input[0] * self.output_layers.neurons[0].calculate_pd_total_net_input_wrt_weiths(w_ho)

                    self.output_layer.neurons[0].weights[w_ho] -= self.LEARNING_RATE*pd_error_wrt_weights

                # update hidden neuron weights

            for h in range(len(self.hidden_layer.neurons)):
                for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                    pd_errors_wrt_output_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weights(w_ih)

                    self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight


        def calculate_total_error(self, training_sets):
            total_error = 0
            for t in range(len(training_sets)):
                training_inputs, training_outputs = training_sets[t]
                self.feed_forward(training_inputs)
                for o in range(len(training_outputs)):
                    total_error += self.output_layer.neurons[o].calculate_error(training_outputs[0])

                return total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):
        # every neuron is a layer share the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(neuron(self.bias)) # going to fix this error after some time, but now let's move forward

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for i in range(len(self.neurons)):
            print('Neuron', n)
            for w in range (len(self.neurons[n].weights)):
                print('weight', self.neurons[n].weights[w])
            print ('Bias', self.bias)

    def feed_forwards(self, inputs):
        outputs = []
        for neurons in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_output(self):
        outputs = []
        for neurons in self.neurons:
            otuputs.append(neuron.output)
        return outputs

    class Neuron
        def __init__(self, bias):
            self.bias = bias
            self.weights = []


        def calculate_outputs(self, inputs):
            self.inputs = inputs
            self.outputs = self.squash(self.caclulate_total_net_input())
            return self.outputs

        def calculate_total_net_input(self):
            total = 0
            for i in range(len(self.inputs)):
                total += self.inputs[i] * self.weights[i]
            return total + self.bias

        # apply the logistics function to squash the output of the neurob
        # the tesult is sometimes referred to as net[2

        def squash(self, total_net_input):
            return 1/ (1 + math.exp(total_net_input))


        def calculate_pd_error_wrt_total_net_input(self, target_output:):
            return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();


        def calculate_error(self, target_output):
            return 0.5 * (target_output - self.output) ** 2


















