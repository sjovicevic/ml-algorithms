class Layer:

    def __init__(self, n_neurons, activation_f):
        self.n_neurons = n_neurons
        self.activation_f = activation_f

    def pisi(self, x):
        print(f'Ovo je x: {x}')


def nebitno():
    print('ovo je nebitno')


l1 = Layer(10, nebitno)
l2 = Layer(5, nebitno)
l3 = Layer(3, nebitno)
layer_list = [l1, l2, l3]

print(layer_list.__getitem__(1))
