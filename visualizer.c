#include "neuralNetwork.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RANDOMSEED 839459
#define ITERATIONS 1000000
#define learning_rate 0.1

int main(int argc, char **argv)
{
	srand(RANDOMSEED); /*Layer *layer1 = create_layer(2, 1, sigmoid, sigmoid_prime);
	Layer *layer2 = create_layer(1, 0, sigmoid, sigmoid_prime);
	Layer *network[] = {layer1, layer2};

	set_layer_values(layer1, (double[]){1, 1});
	print_layer_values(layer1);
	print_layer_values(layer2);
	forward_propagate(network, 2);
	print_layer_values(layer1);
	print_layer_values(layer2);
	printf("%f, %f\n", layer1->weights[0][0], layer1->weights[1][0]);*/
	Layer *layer = create_layer(2, 3, sigmoid, sigmoid_prime);
	Layer *layer2 = create_layer(3, 1, sigmoid, sigmoid_prime);
	Layer *layer3 = create_layer(1, 0, sigmoid, sigmoid_prime);

	int number_of_layers = 3;
	Layer *layers[] = {layer, layer2, layer3};

	for (int i = 0; i < ITERATIONS; i++)
	{
		if (verbose)
			printf("1, 0: \n");
		set_layer_values(layer, (double[]){1, 0});
		forward_propagate(layers, number_of_layers);
		train_network(layers, number_of_layers, learning_rate, (double[]){1});

		if (verbose)
			printf("1, 1: \n");
		set_layer_values(layer, (double[]){1, 1});
		forward_propagate(layers, number_of_layers);
		train_network(layers, number_of_layers, learning_rate, (double[]){0});

		if (verbose)
			printf("0, 0: \n");
		set_layer_values(layer, (double[]){0, 0});
		forward_propagate(layers, number_of_layers);
		train_network(layers, number_of_layers, learning_rate, (double[]){0});

		if (verbose)
			printf("0, 1: \n");
		set_layer_values(layer, (double[]){0, 1});
		forward_propagate(layers, number_of_layers);
		train_network(layers, number_of_layers, learning_rate, (double[]){1});
	}
	set_layer_values(layer, (double[]){0, 1});
	forward_propagate(layers, number_of_layers);
	print_layer_values(layer3);

	set_layer_values(layer, (double[]){1, 1});
	forward_propagate(layers, 3);
	print_layer_values(layer3);
}
