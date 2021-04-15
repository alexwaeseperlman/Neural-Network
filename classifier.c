#include "neuralNetwork.h"
#include "util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#define RANDOMSEED 839459
#define ITERATIONS 20
#define learning_rate 0.01

void print_mnist(double *mnist_data) {
	for (int x = 0; x < 28; x++) {
		for (int y = 0; y < 28; y++) {
			if (mnist_data[x * 28 + y] > 0.75)
				printf("▓▓");
			else if (mnist_data[x * 28 + y] > 0.35)
				printf("▒▒");
			else
				printf("░░");
		}
		printf("\n");
	}
}

void XOR() {
	Layer *layer1 = create_feedforward_layer(2, sigmoid, sigmoid_prime);
	Layer *layer2 = create_feedforward_layer(3, sigmoid, sigmoid_prime);
	Layer *layer3 = create_feedforward_layer(1, sigmoid, sigmoid_prime);

	connect_all_layers((Layer *[]){layer1, layer2, layer3}, 3);

	double **datasetX = malloc(4 * sizeof(double *));
	datasetX[0] = malloc(sizeof(double) * 2);
	datasetX[0][0] = 0.0;
	datasetX[0][1] = 0.0;
	datasetX[1] = malloc(sizeof(double) * 2);
	datasetX[1][0] = 1.0;
	datasetX[1][1] = 1.0;
	datasetX[2] = malloc(sizeof(double) * 2);
	datasetX[2][0] = 0.0;
	datasetX[2][1] = 1.0;
	datasetX[3] = malloc(sizeof(double) * 2);
	datasetX[3][0] = 1.0;
	datasetX[3][1] = 0.0;

	double **datasetY = malloc(4 * sizeof(double *));
	datasetY[0] = malloc(sizeof(double));
	datasetY[0][0] = 0.0;
	datasetY[1] = malloc(sizeof(double));
	datasetY[1][0] = 0.0;
	datasetY[2] = malloc(sizeof(double));
	datasetY[2][0] = 1.0;
	datasetY[3] = malloc(sizeof(double));
	datasetY[3][0] = 1.0;

	int number_of_layers = 3;

	train_on_dataset(layer1, datasetX, datasetY, 4, ITERATIONS, learning_rate);
	set_layer_values(layer1, (double[]){0.0, 0.0});
	forward_propagate_layer(layer1);
	printf("0.0, 0.0: ");
	print_layer_values(find_last_layer(layer1));
}
int main(int argc, char **argv) {
	srand(RANDOMSEED);
	/*Layer *layer1 = create_feedforward_layer(2, 1, sigmoid, sigmoid_prime);
	Layer *layer2 = create_feedforward_layer(1, 0, sigmoid, sigmoid_prime);
	Layer *network[] = {layer1, layer2};

	set_layer_values(layer1, (double[]){1, 1});
	print_layer_values(layer1);
	print_layer_values(layer2);
	forward_propagate_layer(network, 2);
	print_layer_values(layer1);
	print_layer_values(layer2);
	printf("%f, %f\n", layer1->weights[0][0], layer1->weights[1][0]);*/
	
	Layer *input = create_reshape_layer(784, 1, 1, 1);
	Layer *layer1 = create_reshape_layer(28, 28, 1, 3);
	Layer *layer2 = create_convolution_layer(2, 2, 5, 5, 2, relu, relu_prime);
	Layer *layer3 = create_convolution_layer(2, 2, 3, 3, 4, relu, relu_prime);
	Layer *layer4 = create_feedforward_layer(196, relu, relu_prime);
	Layer *output = create_feedforward_layer(10, sigmoid, sigmoid_prime);
	connect_layers(input, layer1);
	connect_layers(layer1, layer2);
	connect_layers(layer2, layer3);
	connect_layers(layer3, layer4);
	connect_layers(layer4, output);/**/
	
	/*Layer *layer1 = create_feedforward_layer(784, sigmoid, sigmoid_prime);
	Layer *layer2 = create_feedforward_layer(20, sigmoid, sigmoid_prime);
	//Layer *layer3 = create_feedforward_layer(32, sigmoid, sigmoid_prime);
	Layer *output = create_feedforward_layer(10, sigmoid, sigmoid_prime);
	connect_layers(layer1, layer2);
	connect_layers(layer2, output);//*/ /**/
	summarizeNetwork(input);
	mnist_data *data;
	unsigned int cnt;
	int ret;

	double labels[10][10] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
	                         {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	                         {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
	                         {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
	                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

	if (ret = mnist_load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", &data, &cnt)) {
		printf("An error occured: %d\n", ret);
	} else {
		printf("image count: %d\n", cnt);

		double **datasetX = malloc(cnt * sizeof(double *));
		for (int i = 0; i < cnt; i++) {
			datasetX[i] = malloc(784 * sizeof(double));

			for (int x = 0; x < 28; x++) {
				for (int y = 0; y < 28; y++) {
					datasetX[i][x * 28 + y] = data[i].data[x][y];
				}
			}
		}
		double **datasetY = malloc(cnt * sizeof(double *));
		for (int i = 0; i < cnt; i++) {
			datasetY[i] = labels[data[i].label];
		}

		train_on_dataset(input, datasetX, datasetY, cnt, ITERATIONS, learning_rate);
		printf("done training\n");
		mnist_data *test_data;

		if (ret = mnist_load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", &test_data, &cnt)) {
			printf("Unable to load test dataset\n");
		} else {
			double **testDatasetX = malloc(cnt * sizeof(double *));
			for (int i = 0; i < cnt; i++) {
				testDatasetX[i] = malloc(784 * sizeof(double));

				for (int x = 0; x < 28; x++) {
					for (int y = 0; y < 28; y++) {
						testDatasetX[i][x * 28 + y] = data[i].data[x][y];
					}
				}
			}
			double **testDatasetY = malloc(cnt * sizeof(double *));
			for (int i = 0; i < cnt; i++) {
				testDatasetY[i] = labels[data[i].label];
			}

			while (1) {
				int imageId;
				printf("Enter an image to convert to vector: ");
				scanf("%d", &imageId);
				print_mnist(testDatasetX[imageId]);
				set_layer_values(input, testDatasetX[imageId]);
				forward_propagate_layer(input);
				printf("Vector:\n");
				print_layer_values(output);
				printf("\n\n\n");
			}
			printf("\n");
			free(data);
		}
	}
}
