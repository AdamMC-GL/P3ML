import neuronlayer


class Neuronnetwork:
    """A network of neurons, consists of multiple layers of neurons.
    A layer consists out of a set of neurons. """

    def __init__(self, layers):
        """Initializes all variables when the class is made.
        Contains the list of layers (a set of neurons) which is the network itself"""
        self.pnetwork = []
        for layer in layers:
            self.pnetwork.append(neuronlayer.Neuronlayer(layer))

    def feed_forward(self, inputs):
        """Given an input, gives the output of the network.
        The input is used to calculate the output of the first layer, and those
        outputs are used as the input for the next until the last layer's output is calculated and returned"""
        for layer in self.pnetwork:
            inputs = layer.activate(inputs)  # each input of a layer gives an output that becomes the input of the next layer
        return inputs

    def empty_network(self, layer_amounts, first_layer_inputs=2):
        """Creates an empty network, a network with all weights and biases set on 0.
        The layer_amount parameter is a list of numbers, each number representing the amount of
        neurons on each layer. [2, 3, 2] = 2 neurons on the first layer, 3 on the second and 2 on the last.
        The first layer has 2 inputs by default, all other layers have inputs the same as the neuron amount of the previous layer."""

        self.pnetwork = []
        weights_amount = first_layer_inputs  # Initial amount of inputs for the first layer
        for amount in layer_amounts:
            self.pnetwork.append(neuronlayer.Neuronlayer([[[0] * weights_amount, 0]] * amount))
            weights_amount = amount

    def __str__(self):
        """Returns a string that tells the information of the whole network.
        Gives the bias and weights of each neuron of each layer and
        shows which neuron of which layer it is"""
        string = ""
        count = 0
        for layer in self.pnetwork:
            count += 1
            string += "Layer " + str(count) + ": \n"
            string += str(layer)
        return string


if __name__ == "__main__":
    xor_port = Neuronnetwork([[[[-5, -5], 8], [[8, 8], -1.5]],
                                 [[[6, 6], -9.5]]])  # nand, or, and

    print("XOR port: ")
    input_combinations = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in input_combinations:
        print(xor_port.feed_forward(i)[0])

    assert xor_port.feed_forward([0, 0])[0] < 0.1, "Should be below 0.1"
    assert xor_port.feed_forward([0, 1])[0] > 0.9, "Should be above 0.9"
    assert xor_port.feed_forward([1, 0])[0] > 0.9, "Should be above 0.9"
    assert xor_port.feed_forward([1, 1])[0] < 0.1, "Should be below 0.1"

    print("Half adder: ")
    halfadder = Neuronnetwork([[[[-5, -5], 8], [[8, 8], -1.5]],
                                   [[[6, 6], -9.5], [[-7, 2], 2]]])  # nand, or, and, custom neuron

    for i in input_combinations:
        print(halfadder.feed_forward(i))

    assert halfadder.feed_forward([0, 0])[0] < 0.1 and halfadder.feed_forward([0, 0])[1] < 0.1, "Should be both below 0.1"
    assert halfadder.feed_forward([0, 1])[0] > 0.9 and halfadder.feed_forward([0, 1])[1] < 0.1, "First should be above 0.9 and second should be below 0.1"
    assert halfadder.feed_forward([1, 0])[0] > 0.9 and halfadder.feed_forward([1, 0])[1] < 0.1, "First should be above 0.9 and second should be below 0.1"
    assert halfadder.feed_forward([1, 1])[0] < 0.1 and halfadder.feed_forward([1, 1])[1] > 0.9, "First should be below 0.1 and second should be above 0.9"


    print("Empty network of 3 layers: ")
    # showcasing empty_network() and __str__()
    testnet = Neuronnetwork([])
    testnet.empty_network([2, 3, 2])
    print(testnet)
