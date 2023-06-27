import numpy as np

layers = [120, 20, 10, 5, 3]

input_layer = layers[0]
hidden_layer = layers[1:-1:1]
output_layer = layers[-1]
n_features = 4
parameters = {}

for layer, index in zip(layers, range(len(layers))):
    if index == 0:
        parameters[f'W{index}'] = np.random.uniform(low=-1, high=1, size=(n_features, layer))
        previous_layer_number = layer
        continue

    parameters[f'W{index}'] = np.random.uniform(low=-1, high=1, size=(previous_layer_number, layer))
    parameters[f'b{index}'] = 0
    previous_layer_number = layer

for i in range(len(parameters) // 2):
    print(parameters[f'W{i}'].shape)
