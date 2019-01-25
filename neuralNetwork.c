#include "neuralNetwork.h"
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"

Layer *EMPTY_LAYER;
void initializeNN() {
	EMPTY_LAYER = create_layer(0, linear, linear_prime);
}

// Create a layer
Layer *create_layer(int neurons, double (*activation)(double), double (*activation_prime)(double)) {
	Layer *out = malloc(sizeof(Layer));
	if (!out) {
		perror("malloc Layer");
		exit(1);
	};
	out->output = EMPTY_LAYER;
	out->input = EMPTY_LAYER;
	out->activation = activation;
	out->activation_prime = activation_prime;
	out->neurons = neurons;
	// Allocate memory for each part of the layer
	out->values = malloc(neurons * sizeof(double));
	out->error = malloc(neurons * sizeof(double));
	out->bias = malloc(neurons * sizeof(double));

	// This is a two dimensional array which means that it needs to allocate each element in a loop
	out->weights = malloc(neurons * sizeof(double *));
	if (!(out->values && out->bias && out->weights)) {
		perror("Failed to allocate memory");
		exit(1);
	}
	for (int i = 0; i < neurons; i++) {
		out->values[i] = 0;
		out->error[i] = 0;
		out->bias[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
	}
	return out;
};

// Connect two layers and initialize the weights randomly
void connect_layers(Layer *input, Layer *output) {
	for (int i = 0; i < input->neurons; i++) {
		// Allocate the weights for each neuron
		input->weights[i] = malloc(output->neurons * sizeof(double));
		for (int j = 0; j < output->neurons; j++) {
			// Set the weights to random
			input->weights[i][j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
		}
	}
	input->output = output;
	output->input = input;
}

// Connect all of the inputted layers
void connect_all_layers(Layer **layers, int number_of_layers) {
	for (int i = 1; i < number_of_layers; i++) {
		connect_layers(layers[i - 1], layers[i]);
	}
}

// Find the last layer in a series of connected layers
Layer *find_last_layer(Layer *firstLayer) {
	if (firstLayer->output == EMPTY_LAYER) {
		return firstLayer;
	}
	return find_last_layer(firstLayer->output);
}

// Set the values of a layer
void set_layer_values(Layer *layer, double values[]) {
	for (int i = 0; i < layer->neurons; i++) {
		layer->values[i] = values[i];
	}
};

// Reset the values of a layer
void reset_values_layer(Layer *layer) {
	for (int i = 0; i < layer->neurons; i++) {
		layer->values[i] = 0;
	}
}
// Reset the values of a layer and all of it's outputs
void reset_values_recursive_layer(Layer *layer) {
	reset_values_layer(layer);
	if (layer->output != EMPTY_LAYER) {
		reset_values_recursive_layer(layer->output);
	}
}

// Reset the error values of a layer
void reset_errors_layer(Layer *layer) {
	for (int i = 0; i < layer->neurons; i++) {
		layer->error[i] = 0;
	}
}
// Reset the error values of a layer and all of it's outputs
void reset_errors_recursive_layer(Layer *layer) {
	reset_errors_layer(layer);
	if (layer->output != EMPTY_LAYER) {
		reset_errors_recursive_layer(layer->output);
	}
}

// Forward propagates a layers values to its output layer
void activate_layer(Layer *layer) {
	if (layer->output == EMPTY_LAYER) {
		fprintf(stderr, "Unable to activate network: output is EMPTY_LAYER\n");
		return;
	}
#pragma omp parallel for
	for (int weight = 0; weight < layer->output->neurons; weight++) {
		layer->output->values[weight] = 0;
		for (int neuron = 0; neuron < layer->neurons; neuron++) {
			layer->output->values[weight] += layer->weights[neuron][weight] * layer->values[neuron];
		}
		layer->output->values[weight] = layer->output->activation(layer->output->values[weight]);
	}
}

// Activate layers recursively
void forward_propagate_layer(Layer *firstLayer) {
	if (firstLayer->output != EMPTY_LAYER) {
		activate_layer(firstLayer);
		forward_propagate_layer(firstLayer->output);
	}
}

// Back propagates the error from the inputted layer to it's input layer
void back_activate_error_layer(Layer *layer) {
	if (layer->input == EMPTY_LAYER) {
		fprintf(stderr, "Unable to back activate network: input is EMPTY_LAYER\n");
		return;
	}
#pragma omp parallel for
	for (int neuron = 0; neuron < layer->input->neurons; neuron++) {
		layer->input->error[neuron] = 0;
		for (int weight = 0; weight < layer->neurons; weight++) {
			layer->input->error[neuron] += layer->input->weights[neuron][weight] * layer->error[weight];
		}
		layer->input->error[neuron] *= layer->input->activation_prime(layer->input->values[neuron]);
	}
}
// Propagates the error from the inputed layer recursively until it gets to EMPTY_LAYER
void back_propagate_error_layer(Layer *lastLayer) {
	back_activate_error_layer(lastLayer);
	if (lastLayer->input->input != EMPTY_LAYER) {
		back_propagate_error_layer(lastLayer->input);
	}
}

void update_weights_layer(Layer *outputLayer, double learning_rate) {
	if (outputLayer->input == EMPTY_LAYER) {
		fprintf(stderr, "Unable to update weights: input is EMPTY_LAYER\n");
		return;
	}
	// Start by updating the bias
	for (int i = 0; i < outputLayer->neurons; i++) {
		outputLayer->bias[i] += learning_rate * outputLayer->error[i];
	}

	// Update all the weights in the previous layer
	for (int i = 0; i < outputLayer->input->neurons; i++) {
		for (int j = 0; j < outputLayer->neurons; j++) {
			outputLayer->input->weights[i][j] += learning_rate * outputLayer->error[j] * outputLayer->input->values[i];
		}
	}
}

// Update the weights of an entire network based on a list of target outputs
void train_layers(Layer *lastLayer, double learning_rate, double targetOutputs[]) {
	// Generate the error for the last layer based on the target outputs
	for (int i = 0; i < lastLayer->neurons; i++) {
		lastLayer->error[i] = (targetOutputs[i] - lastLayer->values[i]) * lastLayer->activation_prime(lastLayer->values[i]);
	}

	// Back propagate the generated error
	back_propagate_error_layer(lastLayer);

	// Update weights of the entire network
	Layer *cursor = lastLayer;
	while (cursor->input != EMPTY_LAYER) {
		update_weights_layer(cursor, learning_rate);
		cursor = cursor->input;
	}
}
void train_on_dataset(Layer *firstLayer, double **x, double **y, int number_of_records, int epochs,
                      double learning_rate) {
	int counter = 0;
	Layer *lastLayer = find_last_layer(firstLayer);
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < number_of_records; j++) {
			if (counter % 100 == 0) printf("%d iterations done.\n", counter);
			counter++;
			set_layer_values(firstLayer, x[j]);
			forward_propagate_layer(firstLayer);
			train_layers(lastLayer, learning_rate, y[j]);
		}
	}
}

// Print the values of a layer
void print_layer_values(Layer *layer) {
	for (int i = 0; i < layer->neurons; i++) {
		printf("%f, ", layer->values[i]);
	}
}

void print_network(Layer **inputLayers, int number_of_layers) {
	for (int i = 0; i < number_of_layers; i++) {
		print_layer_values(inputLayers[i]);
	}
}