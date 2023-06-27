layers = [120, 20, 10, 5, 3]

input_layer = layers[0]
hidden_layer = layers[1:-1:1]
output_layer = layers[-1]

parameters = {}
for layer, index in zip(layers, range(len(layers))):
    parameters[f'W{index}'] = 1
    parameters[f'b{index}'] = 0

print(parameters)
print(input_layer)
print(hidden_layer)
print(output_layer)