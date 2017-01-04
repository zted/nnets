import numpy as np
import activations as A

class Layer(object):
    def __init__(self, output_dim, input_dim, activation='tanh', layer_type='hidden'):
        activations = {'linear':(A.linear, A.linearPrime), 'relu':(A.relu, A.reluPrime),
                       'sigmoid':(A.sigmoid, A.sigmoidPrime), 'tanh':[A.tanh, A.tanhPrime]}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.rand(input_dim, output_dim)
        # the +1 is for bias
        assert activation in activations.keys()
        self.activation = activations[activation][0]
        self.activationPrime = activations[activation][1]
        self.attr = 'Layer'
        self.matrix_product = None
        self.delta = None
        self.input = None
        self.delta_next = None
        self.gradient = None
        self.output = None
        self.learning_rate = 0.5
        assert layer_type in ['first','last','hidden']
        self.layer_type = layer_type

    def output_dim(self):
        return self.output_dim

    def compute_output(self, input):
        self.input = input
        # fix this later, only hidden layers have biases
        self.matrix_product = np.dot(input, self.weights)
        self.output = self.activation(self.matrix_product)
        return self.output

    def compute_delta(self, delta_next, weights_next=None):
        self.delta_next = delta_next
        if self.layer_type == 'last':
            self.delta = self.activationPrime(self.matrix_product) * delta_next
        else:
            self.delta = np.dot(delta_next, weights_next.T) * self.activationPrime(self.matrix_product)
        return self.delta

    def compute_gradient(self):
        self.gradient = np.dot(self.delta.T, self.input)
        return self.gradient

    def update_weight(self):
        self.weights += self.learning_rate*self.gradient.T
        return


class Network(object):
    def __init__(self):
        self.layers = []
        self.attr = 'Network'

    def addLayer(self, layer):
        assert layer.attr == 'Layer'
        self.layers.append(layer)

    def compute_output(self, input):
        for l in self.layers:
            out = l.compute_output(input)
            input = out
        return out

    def compute_error(self, input_x, input_y):
        predicted_y = self.compute_output(input_x)
        diff = input_y - predicted_y
        return diff


a = Layer(6, 5, layer_type='first')
b = Layer(3, 6, layer_type='last')
x = np.array([[0.7, 0.2, -0.5, 0.1, 0.5], [0.3, -0.2, 0.3, -0.7, 0.4]])
y = np.array([[1.0, 0.0, 0.0],[1.0, 1.0, 0.0]])

model = Network()
model.addLayer(a)
model.addLayer(b)
for i in range(0, 20):
    ans = model.compute_output(x)
    if i == 0:
        print "First answer: {}".format(ans)
    error = model.compute_error(x, y)
    b_delta = b.compute_delta(error)
    b_gradient = b.compute_gradient()
    a_delta = a.compute_delta(b_delta, b.weights)
    a_gradient = a.compute_gradient()
    a.update_weight()
    b.update_weight()
    print np.sum(error*error)
print ans