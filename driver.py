from typing_extensions import Annotated
import neural_net as ann
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

layer_sizes = (784, 128, 10)

net = ann.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images, training_labels)
