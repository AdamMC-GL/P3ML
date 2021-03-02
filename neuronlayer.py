import neuron


class Neuronlayer:
    """The neuron layer class. Contains a list of neurons classes."""

    def __init__(self,  neurons):
        """Initializes all variables when the class is made.
        Contains the list of layers (a set of neurons) which is the network itself"""
        self.layer = []
        self.neuron_amount = len(neurons)
        for i in neurons:
            self.layer.append(neuron.Neuron(i[0], i[1]))  # i[0] are the weights of each neuron, i[1] are the biases

    def activate(self, inputs):
        """Given an input, gives the output of the network.
        The input is used to calculate the output of each neuron until you
        have a list of all the outputs of the layer."""
        outputs = []
        for neuron in self.layer:
            outputs.append(neuron.activate(inputs))
        return outputs

    def __str__(self):
        """Rerurns a string that tell the information of a whole layer
        Gives the bias and weights of each neuron of the layer and
        shows which neuron it is"""
        stri = ""
        count = 0
        for i in self.layer:
            count += 1
            stri += "neuron " + str(count) + ": " + str(i) + "\n"
        return stri


if __name__ == "__main__":
    a = Neuronlayer([[[1, 2], 2],
                     [[1, 2], 3],
                     [[1, 2], 1],
                     [[1, 2], 2],
                     [[1, 2], 3]])
    print(a)
