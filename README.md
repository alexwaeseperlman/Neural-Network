# Convolutional Neural Network in C

## What is this?
This is a convolutional neural network written entirely from scratch in C. I wrote it as an excercise to solidify my understanding of neural networks. It achieves an accuracy of 82% on the test dataset of MNIST.

It supports fully connected and convolutional layers with any activation function. It's trained using back propagation and stochastic gradient descent. The only loss function supported is mean squared error which isn't ideal for classification, but it's still able to train. If I were to do it again I would support the use of custom loss functions and use cross-entropy for mnist. 
