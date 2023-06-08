import numpy as np
"""
# way of doing things
for targ_idx, distribution in zip(class_targets, softmax_output):
    print(distribution[targ_idx])

# other way of doing things
neg_log = -np.log(softmax_output[range(len(softmax_output)), class_targets])
"""

softmax_output = np.array([
                        [0.7, 0.1, 0.2],
                        [0.1, 0.5, 0.4],
                        [0.02, 0.9, 0.08]])

# one hot indices
class_targets = np.array([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0]])

if len(class_targets.shape) == 1:
    correct_confidences = softmax_output[range(len(softmax_output)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_output * class_targets, axis=1)

neg_log = -np.log(correct_confidences)
average_mean = np.mean(neg_log)
print(average_mean)
