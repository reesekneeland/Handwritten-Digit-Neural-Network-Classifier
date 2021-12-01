from typing_extensions import Annotated
import neural_net as ann
import numpy as np
import matplotlib as plt

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    validation_images = data['validation_images']
    validation_labels = data['validation_labels']

layer_sizes = (784, 128, 10)

net = ann.NeuralNetwork(layer_sizes)
net.train(training_images, training_labels, validation_images, validation_labels)
net.test(test_images, test_labels)
net.print_accuracy()
