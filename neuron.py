import math
class Neuron:
    """The class of a neuron, contains a set of weights and a bias.
    The function activate() dictates if the neuron has a True or False output determined
    by its weights and bias."""

    def __init__(self, weights=None, bias=None):
        """Initializes all variables when the class is made.
        Contains the weights of the neuron and its bias, the
        output of the neuron always starts as false"""
        self.weights = weights
        self.bias = bias
        self.output = 0

    def activate(self, inputs):
        """Given the input this function calcualtes wether or not the neuron
        should activate. Each weight with input True is summed up plus the bias,
        if this is greater than or equal to 0 the neuron activates"""
        active_weights = [self.weights[i] * inputs[i] for i in
                          range(len(self.weights))]  # A list of all the weights where the input is True
        self.output = 1 / (1 + math.exp(-(sum(active_weights) + self.bias)))

        return self.output

    def __str__(self):
        """Rerurns a string that tell the information of the neuron
        Gives the bias and weights of the neuron."""
        return "bias: " + str(self.bias) + ", weights: " + str(self.weights)


if __name__ == "__main__":
    input_combinations = [[0, 0], [0, 1], [1, 0], [1, 1]]

    invert = Neuron([-7], 4)
    print("Invert: ")
    print(invert.activate([0]))
    print(invert.activate([1]))

    assert invert.activate([0]) > 0.9, "Should be above 0.9"
    assert invert.activate([1]) < 0.1, "Should be below 0.1"

    andport = Neuron([5, 5], -7.5)
    print("AND: ")
    for i in input_combinations:
        print(andport.activate(i))

    assert andport.activate([0, 0]) < 0.1, "Should be below 0.1"
    assert andport.activate([0, 1]) < 0.1, "Should be below 0.1"
    assert andport.activate([1, 0]) < 0.1, "Should be below 0.1"
    assert andport.activate([1, 1]) > 0.9, "Should be above 0.9"

    orport = Neuron([5, 5], -2.5)
    print("OR: ")
    for i in input_combinations:
        print(orport.activate(i))

    assert orport.activate([0, 0]) < 0.1, "Should be below 0.1"
    assert orport.activate([0, 1]) > 0.9, "Should be above 0.9"
    assert orport.activate([1, 0]) > 0.9, "Should be above 0.9"
    assert orport.activate([1, 1]) > 0.9, "Should be above 0.9"

    norport = Neuron([-5, -5, -5], 2.5)
    print("NOR: ")
    input_combinations_3 = [[0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 1, 1]]
    for i in input_combinations_3:
        print(norport.activate(i))

    assert norport.activate([0, 0, 0]) > 0.9, "Should be above 0.9"
    assert norport.activate([0, 1, 1]) < 0.1, "Should be below 0.1"
    assert norport.activate([0, 1, 0]) < 0.1, "Should be below 0.1"
    assert norport.activate([1, 1, 1]) < 0.1, "Should be below 0.1"
