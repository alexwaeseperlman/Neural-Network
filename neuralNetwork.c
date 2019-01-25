#include "neuralNetwork.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"

// Create a layer
Layer *create_layer(int neurons, int outputNeurons, double (*activation)(double), double (*activation_prime)(double)) {
	Layer *out = malloc(sizeof(Layer));
	if (!out) {
		perror("malloc Layer");
		exit(1);
	};
	out->activation = activation;
	out->activation_prime = activation_prime;
	out->neurons = neurons;
	out->output_neurons = outputNeurons;
	// Allocate memory for each part of the layer
	out->values = malloc(neurons * sizeof(double));
	out->bias = malloc(neurons * sizeof(double));

	// This is a two dimensional array which means that it needs to allocate each element in a loop
	out->weights = malloc(neurons * sizeof(double *));
	if (!(out->values && out->bias && out->weights)) {
		perror("Failed to allocate memory");
		exit(1);
	}
	for (int i = 0; i < neurons; i++) {
		out->values[i] = 0;
		out->bias[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
		// Allocate the weights for each neuron
		out->weights[i] = malloc(outputNeurons * sizeof(double));
		for (int j = 0; j < outputNeurons; j++) {
			// Set the weights to random
			out->weights[i][j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
		}
	}
	return out;
};

// Set the values of a layer
void set_layer_values(Layer *layer, double values[]) {
	for (int i = 0; i < layer->neurons; i++) {
		layer->values[i] = values[i];
	}
};

void reset_values(Layer **layers, int number_of_layers) {
	for (int i = 0; i < number_of_layers; i++) {
		for (int j = 0; j < layers[i]->neurons; j++) {
			layers[i]->values[j] = 0;
		}
	}
}

// Forward propagate a list of layers to other layers
void forward_propagate(Layer *layers[], int number_of_layers) {
	if (number_of_layers > 1) {
		// Don't reset the first the layer because it contains all the inputs
		reset_values(layers + 1, number_of_layers - 1);
		// Loop through layers - 1 because the last layer doesn't propagate
		for (int i = 0; i < number_of_layers - 1; i++) {
			// Make sure that the layers are able to be connected
			if (layers[i]->output_neurons == layers[i + 1]->neurons) {
				if (forward_prop_debug) {
					print_layer_values(layers[i]);
					printf("Layer %d: ", i);
				}
#pragma omp parallel for
				for (int weight = 0; weight < layers[i]->output_neurons; weight++) {
					if (forward_prop_debug) printf("activation(%f", layers[i + 1]->values[weight]);
					for (int neuron = 0; neuron < layers[i]->neurons; neuron++) {
						if (forward_prop_debug) printf(" + %f * %f", layers[i]->weights[neuron][weight], layers[i]->values[neuron]);
						layers[i + 1]->values[weight] += layers[i]->weights[neuron][weight] * layers[i]->values[neuron];
					}
					if (forward_prop_debug) printf(" + %f", layers[i + 1]->bias[weight]);
					layers[i + 1]->values[weight] += layers[i + 1]->bias[weight];
					if (forward_prop_debug) printf(" = %f)", layers[i + 1]->values[weight]);
					layers[i + 1]->values[weight] = layers[i + 1]->activation(layers[i + 1]->values[weight]);
					if (forward_prop_debug) printf(" = %f\n", layers[i + 1]->values[weight]);
				}
			} else {
				perror("Number of outputs is not equal to the number of neurons in the next layer");
				exit(1);
			}
		}
	} else {
		printf("Only one layer inputted\n");
	}
};

// Back propagate a layer
double *back_propagate(Layer *inputLayer, Layer *outputLayer, double *delta, double learning_rate) {
	if (verbose) printf("\n");
	// have a place to store the errors in order to calculate delta
	double *errors = malloc(inputLayer->neurons * sizeof(double));
	for (int i = 0; i < inputLayer->neurons; i++) {
		errors[i] = 0;
	}

	// Back propagate the error
	for (int i = 0; i < inputLayer->neurons; i++) {
		for (int j = 0; j < outputLayer->neurons; j++) {
			errors[i] += (inputLayer->weights[i][j] * delta[j]);
		}
		errors[i] *= inputLayer->activation_prime(inputLayer->values[i]);
	}

	// Adjust biases
	for (int i = 0; i < outputLayer->neurons; i++) {
		outputLayer->bias[i] += learning_rate * delta[i];
	}

	// Adjust the weights based on the newly calculated delta
	for (int i = 0; i < inputLayer->neurons; i++) {
		for (int j = 0; j < outputLayer->neurons; j++) {
			if (verbose > 0) {
				printf("Adjusting %d,%d by %f*%f*%f=%f from the input layer\n", i, j, learning_rate, delta[j],
				       inputLayer->values[i], learning_rate * delta[j] * inputLayer->values[i]);
				printf("Old value: %f, ", inputLayer->weights[i][j]);
			}
			inputLayer->weights[i][j] += learning_rate * delta[j] * inputLayer->values[i];

			if (verbose > 0) {
				printf("New value: %f\n", inputLayer->weights[i][j]);
				usleep(100);
			}
		}
	}

	return errors;
}

void train_layers(Layer **layers, int number_of_layers, double learningRate, double targetOutputs[]) {
	double *delta = malloc(layers[number_of_layers - 1]->neurons * sizeof(double));

	for (int i = 0; i < layers[number_of_layers - 1]->neurons; i++) {
		delta[i] = (targetOutputs[i] - layers[number_of_layers - 1]->values[i]) *
		           layers[number_of_layers - 1]->activation_prime(layers[number_of_layers - 1]->values[i]);

		if (verbose)
			printf("Target output: %f, current output: %f, error: %f, delta: %f\n", targetOutputs[i],
			       layers[number_of_layers - 1]->values[i], targetOutputs[i] - layers[number_of_layers - 1]->values[i],
			       delta[i]);
	}
	// Only loop down to 1 because it can't be back propagated further
	for (int i = number_of_layers - 1; i > 0; i--) {
		double *old = delta;
		delta = back_propagate(layers[i - 1], layers[i], delta, learningRate);
		free(old);
	}
	free(delta);
}
void train_on_dataset(Layer **layers, int number_of_layers, double **x, double **y, int number_of_records, int epochs,
                      double learning_rate) {
	int counter = 0;
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < number_of_records; j++) {
			if (counter % 100 == 0) printf("%d iterations done.\n", counter);
			counter++;
			set_layer_values(layers[0], x[j]);
			forward_propagate(layers, number_of_layers);
			train_layers(layers, number_of_layers, learning_rate, y[j]);
		}
	}
}

void print_layer_values(Layer *layer) {
	printf("       ");
	for (int i = 0; i < layer->neurons; i++) {
		printf("%f, ", layer->bias[i]);
	}
	printf("\nLayer: ");

	for (int i = 0; i < layer->neurons; i++) {
		printf("%f, ", layer->values[i]);
	}
	for (int i = 0; i < layer->output_neurons; i++) {
		printf("\n         ");
		for (int j = 0; j < layer->neurons; j++) {
			printf("%f, ", layer->weights[j][i]);
		}
	}
	printf("\n");
}

void print_network(Layer **inputLayers, int number_of_layers) {
	for (int i = 0; i < number_of_layers; i++) {
		print_layer_values(inputLayers[i]);
	}
}