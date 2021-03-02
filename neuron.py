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
        self.output = 0
        if sum(active_weights) + self.bias >= 0:
            self.output = 1

        return 1 / (1 + math.exp(-(sum(active_weights) + self.bias)))

    def error(self, training_set):
        """Given the traning set containing all inputs and expected outputs, this function
        calculates the Mean Squared Error. The formula of MSE is  Σ | d – y |^2 / n, where
        d is the output being predicted, y the predicted output, and n the size of total predictions"""
        sigma = 0
        for data in training_set:  # calculate sigma d - y ^ 2
            inputs = data[0]
            exp_outcome = data[1]  # d
            output = self.activate(inputs)  # calculate y

            sigma += (exp_outcome - output) ** 2  # sum of each d - y ^ 2 instance

        mse = sigma / len(
            training_set)  # each value in training_set contains one expected prediction, meaning that the lengts of the training set is the same as the size of total predictions
        return mse

    def update(self, training_set, epochs=1, learning_rate=1.0):
        """Given the traning set containing all inputs and expected outputs, epochs that determines
        how many times the updating is looped through the training set, and learning rate. This function
        updates the parameters of the neuron towards a more accurate one in line with the expected/needed
        output of the neuron"""
        for current_epoch in range(epochs):
            for data in training_set:  # training set contains the input and expected output
                inputs = data[0]
                exp_outcome = data[1]

                output = self.activate(inputs)  # calculate y
                e = exp_outcome - output  # calculate error, if 0 all other calculations add no change

                for weight_index in range(len(self.weights)):
                    self.weights[weight_index] += (learning_rate * e * inputs[
                        weight_index])  # calculate weight delta for each weight and add for each weight
                self.bias += (learning_rate * e)  # calculate bias delta and add to bias

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
