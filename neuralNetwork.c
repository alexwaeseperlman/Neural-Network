#include "neuralNetwork.h"
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include <signal.h>
double access(double *arr, int x, int y, int z, int w, int h, int d);

int empty_layer_initialized;
Layer *EMPTY_LAYER;
#define LAYER_DIMENSIONS_IDX 0
void initializeNN() {
	EMPTY_LAYER = create_feedforward_layer(0, linear, linear_prime);
	EMPTY_LAYER->metaparams[LAYER_DIMENSIONS_IDX] = 0;
}

#define LAYER_X_DIMENSIONS_IDX 1
#define LAYER_Y_DIMENSIONS_IDX 2
#define LAYER_Z_DIMENSIONS_IDX 3

//#define DEBUG_HIGH
//#define DEBUG

#define FF_META_SIZE 2
#define FF_NEURONS 1
#define FF_TYPE 'F'
// Create a layer
Layer *create_feedforward_layer(int neurons, double (*activation)(double), double (*activation_prime)(double)) {
	if (!empty_layer_initialized) {
		empty_layer_initialized = 1;
		initializeNN();
	}
	Layer *out = malloc(sizeof(Layer));
	if (!out) {
		perror("malloc Layer");
		exit(1);
	}
	out->output = EMPTY_LAYER;
	out->input = EMPTY_LAYER;
	out->activation = activation;
	out->activation_prime = activation_prime;
	out->metaparams = malloc(sizeof(int) * FF_META_SIZE);
	out->metaparams[LAYER_DIMENSIONS_IDX] = 1;
	out->metaparams[FF_NEURONS] = neurons;
	out->metaparams[LAYER_Y_DIMENSIONS_IDX] = 1;
	out->metaparams[LAYER_Z_DIMENSIONS_IDX] = 1;
	// Allocate memory for each part of the layer
	out->values = malloc(neurons * sizeof(double));
	out->error = malloc(neurons * sizeof(double));
	out->bias = malloc(neurons * sizeof(double));
	
	out->activate = activate_feedforward_layer;
	out->back_activate_error = back_activate_error_feedforward_layer;
	out->update_weights = update_weights_feedforward_layer;
	
	out->allocated = 0;
	
	out->type = FF_TYPE;

	if (!(out->values && out->bias && out->error && out->metaparams)) {
		perror("Failed to allocate memory");
		exit(1);
	}
	for (int i = 0; i < neurons; i++) {
		out->values[i] = 0;
		out->error[i] = 0;
		out->bias[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
	}
	return out;
}

#define CONV_META_SIZE 10
#define CONV_KERNEL_W 4
#define CONV_KERNEL_H 5
#define CONV_KERNEL_D 6
#define CONV_KERNEL_COUNT 7
#define CONV_STRIDE_X 8
#define CONV_STRIDE_Y 9
#define CONV_TYPE 'C'

Layer *create_convolution_layer(int strideX, int strideY, int kernelWidth, int kernelHeight, int kernels, double (*activation)(double), double (*activation_prime)(double)) {
	if (!empty_layer_initialized) {
		empty_layer_initialized = 1;
		initializeNN();
	}
	Layer *out = malloc(sizeof(Layer));
	if (!out) {
		perror("malloc Layer");
		exit(1);
	}
	out->output = EMPTY_LAYER;
	out->input = EMPTY_LAYER;
	out->activation = activation;
	out->activation_prime = activation_prime;
	out->metaparams = malloc(CONV_META_SIZE * sizeof(int));
	out->metaparams[LAYER_DIMENSIONS_IDX] = 3;
	//out->metaparams[LAYER_Z_DIMENSIONS_IDX] = kernels;
	// Set size on connection
	out->metaparams[CONV_KERNEL_W] = kernelWidth;
	out->metaparams[CONV_KERNEL_H] = kernelHeight;
	out->metaparams[CONV_KERNEL_COUNT] = kernels;
	
	out->metaparams[CONV_STRIDE_X] = strideX;
	out->metaparams[CONV_STRIDE_Y] = strideY;
	//out->metaparams[CONV_KERNEL_D] = depth;
	// Allocate memory for each part of the layer
	//out->values = malloc(size * sizeof(double));
	//out->error = malloc(size * sizeof(double));
	//out->bias = malloc(size * sizeof(double));
	out->type = CONV_TYPE;
	
	out->allocated = 0;
	
	out->activate = activate_convolutional_layer;
	out->back_activate_error = back_activate_error_convolution_layer;
	out->update_weights = update_weights_convolution_layer;

	// This is a one dimensional array with all of the kernels values
	//out->weights = malloc(kernels * kernelWidth * kernelHeight * depth * sizeof(double));
	return out;
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
	for (int i = 0; i < layer->metaparams[FF_NEURONS]; i++) {
		layer->values[i] = values[i];
	}
}

// Reset the values of a layer
void reset_values_layer(Layer *layer) {
	for (int i = 0; i < layer->metaparams[FF_NEURONS]; i++) {
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
	for (int i = 0; i < layer->metaparams[FF_NEURONS]; i++) {
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
void activate_feedforward_layer(Layer *layer) {
	if (layer->output == EMPTY_LAYER) {
		fprintf(stderr, "Unable to activate network: output is EMPTY_LAYER\n");
		return;
	}
#pragma omp parallel for
	for (int weight = 0; weight < layer->output->metaparams[FF_NEURONS]; weight++) {
		layer->output->values[weight] = 0;
		for (int neuron = 0; neuron < layer->metaparams[FF_NEURONS]; neuron++) {
			layer->output->values[weight] += layer->weights[neuron * layer->output->metaparams[FF_NEURONS] + weight] * layer->values[neuron];
			if (isnan(layer->output->values[weight])) {
				printf("NAN: %f, %f, %f\n", layer->output->values[weight], layer->weights[neuron * layer->output->metaparams[FF_NEURONS] + weight], layer->values[neuron]);
			
				raise(SIGTRAP);				
			}
		}
		layer->output->values[weight] = layer->output->activation(layer->output->values[weight]);
	}
}


void printLayerInfo(Layer *layer) {
	if (layer->type == 'F') {
		
		printf("Feed forward layer: Neurons: %d, weights: %d\n", 
			layer->metaparams[FF_NEURONS], layer->metaparams[FF_NEURONS] * layer->output->metaparams[FF_NEURONS]);
	}
	else if (layer->type == 'C') {
		printf("Convolutional layer: Dimensions: %d, width: %d, height: %d, depth: %d, kernels: %d, kernel width: %d, kernel height: %d, kernel depth: %d, weights: %d\n", 
			layer->metaparams[LAYER_DIMENSIONS_IDX],
			layer->metaparams[LAYER_X_DIMENSIONS_IDX],
			layer->metaparams[LAYER_Y_DIMENSIONS_IDX],
			layer->metaparams[LAYER_Z_DIMENSIONS_IDX],
			layer->metaparams[CONV_KERNEL_COUNT],
			layer->metaparams[CONV_KERNEL_W],
			layer->metaparams[CONV_KERNEL_H],
			layer->metaparams[CONV_KERNEL_D],
			layer->metaparams[CONV_KERNEL_COUNT] * layer->metaparams[CONV_KERNEL_W] * layer->metaparams[CONV_KERNEL_H] * layer->metaparams[CONV_KERNEL_D]);
	}
	else if (layer->type == 'R') {
		printf("Reshape layer: %d->%d. Size: %d\n", layer->metaparams[LAYER_DIMENSIONS_IDX], layer->output->metaparams[LAYER_DIMENSIONS_IDX], layer->metaparams[LAYER_X_DIMENSIONS_IDX] * layer->metaparams[LAYER_Y_DIMENSIONS_IDX] * layer->metaparams[LAYER_Z_DIMENSIONS_IDX]);
	}
}

inline void runKernel(Layer *layer, int forwards) {
	//void applyKernel(int imageWidth, int imageHeight, int imageDepth, int kernelWidth, int kernelHeight, int kernelDepth, int kernels, int strideX, int strideY, double *kernel, double *in, double *out, int forwards, double *error) {
	applyKernel(layer->metaparams[LAYER_X_DIMENSIONS_IDX], 
		layer->metaparams[LAYER_Y_DIMENSIONS_IDX], 
		layer->metaparams[LAYER_Z_DIMENSIONS_IDX], 
		layer->metaparams[CONV_KERNEL_W], 
		layer->metaparams[CONV_KERNEL_H], 
		layer->metaparams[CONV_KERNEL_D], 
		layer->metaparams[CONV_KERNEL_COUNT], 
		layer->metaparams[CONV_STRIDE_X], 
		layer->metaparams[CONV_STRIDE_Y], 
		layer->weights,
		forwards ? layer->values : layer->error, 
		forwards ? layer->output->values : layer->output->error, 
		forwards);
}
void activate_convolutional_layer(Layer *layer) {
	//void applyKernel(int imageWidth, int imageHeight, int imageDepth, int kernelWidth, int kernelHeight, int kernelDepth, int kernels, int strideX, int strideY, double *kernel, double *in, double *out, int forwards, double *error) {
	#ifdef DEBUG
	printf("Conv forward prop real size: %d\n", layer->output->metaparams[LAYER_X_DIMENSIONS_IDX] * layer->output->metaparams[LAYER_Y_DIMENSIONS_IDX] * layer->output->metaparams[LAYER_Z_DIMENSIONS_IDX]);
	#endif
	runKernel(layer, 1);
	for (int i = 0; i < layer->metaparams[CONV_KERNEL_COUNT] * layer->metaparams[CONV_KERNEL_W] * layer->metaparams[CONV_KERNEL_H] * layer->metaparams[CONV_KERNEL_D]; i++) {
		layer->values[i] = layer->activation(layer->values[i]);
	}
}
// Activate layers recursively
void forward_propagate_layer(Layer *firstLayer) {
	if (firstLayer->output != EMPTY_LAYER) {
		#ifdef DEBUG	
		printf("%d %c -> %d %c\n", firstLayer->metaparams[LAYER_DIMENSIONS_IDX], firstLayer->type, firstLayer->output->metaparams[LAYER_DIMENSIONS_IDX], firstLayer->output->type);
		#endif
		//print_layer_values(firstLayer);
		firstLayer->activate(firstLayer);
		forward_propagate_layer(firstLayer->output);
	}
	else {
		//printf("Done forward\n");
	}
}

// Back propagates the error from the inputted layer to it's input layer
void back_activate_error_feedforward_layer(Layer *layer) {
	if (layer->output == EMPTY_LAYER) {
		fprintf(stderr, "Unable to back activate network: output is EMPTY_LAYER. No error to back activate\n");
		return;
	}
	#ifdef DEBUG_HIGH
	for (int i = 0; i < layer->output->metaparams[FF_NEURONS]; i++) {
		printf("%f, ", layer->output->error[i]);
	}
	printf("\n\n");
	#endif
	#pragma omp parallel for
	for (int neuron = 0; neuron < layer->metaparams[FF_NEURONS]; neuron++) {
		layer->error[neuron] = 0;
		for (int weight = 0; weight < layer->output->metaparams[FF_NEURONS]; weight++) {
			layer->error[neuron] += layer->weights[neuron * layer->output->metaparams[FF_NEURONS] + weight] * layer->output->error[weight];
			
			if (isnan(layer->output->error[weight])) {
				printf("NAN ERROR: %f, %f, %f\n", layer->output->error[weight], layer->weights[neuron * layer->output->metaparams[FF_NEURONS] + weight], layer->error[neuron]);
			
				raise(SIGTRAP);				
			}
		}
		layer->error[neuron] *= layer->activation_prime(layer->values[neuron]);
	}
}
// Propagates the error from the inputed layer recursively until it gets to EMPTY_LAYER
void back_propagate_error_layer(Layer *lastLayer) {
	if (lastLayer->output != EMPTY_LAYER) {
		#ifdef DEBUG
		printf("Back activating %c -> %c\n", lastLayer->type, lastLayer->input->type);
		#endif	
		lastLayer->back_activate_error(lastLayer);
		
	}
	if (lastLayer->input != EMPTY_LAYER) {
		back_propagate_error_layer(lastLayer->input);
	}
}

void update_weights_feedforward_layer(Layer *layer, double learning_rate) {
	if (layer->output == EMPTY_LAYER) {
		fprintf(stderr, "Unable to update weights. Layer does not have weights\n");
		return;
	}
	// Start by updating the bias
	for (int i = 0; i < layer->metaparams[FF_NEURONS]; i++) {
		layer->bias[i] += learning_rate * layer->error[i];
	}

	// Update all the weights in the previous layer
	for (int i = 0; i < layer->metaparams[FF_NEURONS]; i++) {
		for (int j = 0; j < layer->output->metaparams[FF_NEURONS]; j++) {
			//printf("Weight: %f, Error: %f, value: %f\n", layer->weights[i * layer->output->metaparams[FF_NEURONS] + j], layer->output->error[j], layer->output->values[i]);
			layer->weights[i * layer->output->metaparams[FF_NEURONS] + j] += learning_rate * layer->output->error[j] * layer->values[i];
			//raise(SIGTRAP);
		}
	}
}

void update_weights_convolution_layer(Layer *layer, double learning_rate) {
	//double *acc_kernel_error = malloc(sizeof(double) * layer->metaparams[CONV_KERNEL_COUNT] * layer->metaparams[CONV_KERNEL_D] * layer->metaparams[CONV_KERNEL_W] * layer->metaparams[CONV_KERNEL_H]);
	int firstLoop = 1;
	for (int x = 0; x < layer->metaparams[LAYER_X_DIMENSIONS_IDX]; x += layer->metaparams[CONV_STRIDE_X]) {
		for (int y = 0; y < layer->metaparams[LAYER_Y_DIMENSIONS_IDX]; y += layer->metaparams[CONV_STRIDE_Y]) {
			// Get the kernel positions
			for (int kx = x - layer->metaparams[CONV_KERNEL_W] / 2, ix = 0; kx < x + layer->metaparams[CONV_KERNEL_W] / 2; kx++, ix++) {
				for (int ky = y - layer->metaparams[CONV_KERNEL_H] / 2, iy = 0; ky < y + layer->metaparams[CONV_KERNEL_H] / 2; ky++, iy++) {
					for (int kz = 0, iz = 0; kz < layer->metaparams[CONV_KERNEL_D]; kz++, iz++) {
						for (int k = 0; k < layer->metaparams[CONV_KERNEL_COUNT]; k++) {
							// Set value
							int kernelIdx = k * layer->metaparams[CONV_KERNEL_W] * layer->metaparams[CONV_KERNEL_D] * layer->metaparams[CONV_KERNEL_H] + iz * layer->metaparams[CONV_KERNEL_W] * layer->metaparams[CONV_KERNEL_H] + iy * layer->metaparams[CONV_KERNEL_W] + ix;
							//if (firstLoop) acc_kernel_error[kernelIdx] = 0;
							layer->weights[kernelIdx] += learning_rate * layer->weights[kernelIdx] * 
								access(layer->output->error, kx, ky, k, layer->metaparams[CONV_KERNEL_W], layer->metaparams[CONV_KERNEL_H], layer->metaparams[CONV_KERNEL_D]) *
								access(layer->values, kx, ky, kz, layer->metaparams[CONV_KERNEL_W], layer->metaparams[CONV_KERNEL_H], layer->metaparams[CONV_KERNEL_D]);
						}
					}
				}
			}
			//firstLoop = 0;
		}
	}
	
}

void back_activate_error_convolution_layer(Layer *layer) {
	runKernel(layer, 0);
	
	for (int i = 0; i < layer->metaparams[CONV_KERNEL_COUNT] * layer->metaparams[CONV_KERNEL_W] * layer->metaparams[CONV_KERNEL_H] * layer->metaparams[CONV_KERNEL_D]; i++) {
		layer->error[i] *= layer->activation_prime(layer->values[i]);
	}
}

// Update the weights of an entire network based on a list of target outputs
void train_layers(Layer *lastLayer, double learning_rate, double targetOutputs[]) {
	// Generate the error for the last layer based on the target outputs
	for (int i = 0; i < lastLayer->metaparams[FF_NEURONS]; i++) {
		lastLayer->error[i] = (targetOutputs[i] - lastLayer->values[i]) * lastLayer->activation_prime(lastLayer->values[i]);
	}
	#ifdef DEBUG
	printf("Found error\n");
	#endif
	// Back propagate the generated error
	back_propagate_error_layer(lastLayer);
	#ifdef DEBUG
	printf("Back propagated from last layer\n");
	#endif
	// Update weights of the entire network
	Layer *cursor = lastLayer;
	while (cursor->input != EMPTY_LAYER) {
		cursor = cursor->input;
		cursor->update_weights(cursor, learning_rate);
	}
}
void train_on_dataset(Layer *firstLayer, double **x, double **y, int number_of_records, int epochs, double learning_rate) {
	int counter = 0;
	Layer *lastLayer = find_last_layer(firstLayer);
	for (int i = 0; i < epochs; i++) {
		for (int j = 0; j < number_of_records; j++) {
			if (counter % 1000 == 0) {
				printf("%d iterations done.\n", counter);
			}
			if (counter % 60000 == 0) printf("current error %f\n", find_error(firstLayer, x, y, number_of_records, firstLayer->metaparams[FF_NEURONS], lastLayer->metaparams[FF_NEURONS]));
			counter++;
			set_layer_values(firstLayer, x[j]);
			forward_propagate_layer(firstLayer);
			#ifdef DEBUG
			printf("Training network\n");
			#endif
			train_layers(lastLayer, learning_rate, y[j]);
		}
	}
}

// Print the values of a layer
void print_layer_values(Layer *layer) {
	for (int i = 0; i < layer->metaparams[FF_NEURONS]; i++) {
		printf("%f, ", layer->values[i]);
	}
}

void print_network(Layer **inputLayers, int number_of_layers) {
	for (int i = 0; i < number_of_layers; i++) {
		print_layer_values(inputLayers[i]);
	}
}

int accessGetIndex(int x, int y, int z, int w, int h, int d) {
	int ax = x > w ? w - 1 : (x < 0 ? 0 : x);
	int ay = y > h ? h - 1 : (y < 0 ? 0 : y);
	int az = z > d ? d - 1 : (z < 0 ? 0 : z);
	
	#ifdef DEBUG_HIGH
	printf("Accessing value at %d,%d,%d with target %d,%d,%d. Limited by %d,%d,%d\n", ax, ay, az, x, y, z, w, h, d);
	#endif
	
	return az * w * h + ay * w + ax;
}

double access(double *arr, int x, int y, int z, int w, int h, int d) {
	return arr[accessGetIndex(x, y, z, w, h, d)];
}

void accessSet(double *arr, int x, int y, int z, int w, int h, int d, double v) {
	arr[accessGetIndex(x, y, z, w, h, d)] = v;
}

void applyKernel(int imageWidth, int imageHeight, int imageDepth, int kernelWidth, int kernelHeight, int kernelDepth, int kernels, int strideX, int strideY, double *kernel, double *in, double *out, int forwards) {
	// If going backwards ensure the input image is set to zero
	if (forwards) {
		int size = (imageWidth / strideX) * (imageHeight / strideY) * kernels;
		#ifdef DEBUG
		printf("Conv forward prop calculated size: %d\n", size);
		#endif
		for (int i = 0; i < size; i++) {
			out[i] = 0;
		}
		#ifdef DEBUG
		printf("Cleared output layer\n");
		#endif
		
	}
	else {
		int size = imageWidth * imageHeight * imageDepth;
		for (int i = 0; i < size; i++) {
			in[i] = 0;
		}
	}
	
	int outWidth = imageWidth / strideX;
	int outHeight = imageHeight / strideY;
	int outDepth = kernels;
	
	for (int x = 0; x < imageWidth; x += strideX) {
		for (int y = 0; y < imageHeight; y += strideY) {
			// Get the kernel positions
			for (int kx = x - kernelWidth / 2, ix = 0; kx < x + kernelWidth / 2; kx++, ix++) {
				for (int ky = y - kernelHeight / 2, iy = 0; ky < y + kernelHeight / 2; ky++, iy++) {
					for (int kz = 0, iz = 0; kz < imageDepth; kz++, iz++) {
						for (int k = 0; k < kernels; k++) {
							#ifdef DEBUG_HIGH
							printf("Accessing kernel #%d out of %d at %d,%d,%d limited by %d,%d,%d\n", k, kernels, ix, iy, iz, kernelWidth, kernelHeight, kernelDepth);
							#endif
							// Set value
							int kernelIdx = k * kernelWidth * kernelDepth * kernelHeight + iz * kernelHeight * kernelWidth + iy * kernelWidth + ix;
							double weight = kernel[kernelIdx];
							#ifdef DEBUG_HIGH
							printf("Accessed\n");
							#endif
							int inIdx = kz * imageWidth * imageHeight + y * imageWidth + x;
							int outIdx = k * outWidth * outHeight + y / strideY * outWidth + x / strideX;
							if (!forwards) {
								in[inIdx] += weight * access(out, x, y, k, outWidth, outDepth, kernels);
							}
							else {
								#ifdef DEBUG_HIGH
								printf("Summing kernel at %d,%d,%d limited by %d,%d,%d with stride %d,%d\n", x / strideX, y / strideY, k, outWidth, outHeight, kernels, strideX, strideY);
								#endif
								out[outIdx] += weight * access(in, kx, ky, kz, imageWidth, imageHeight, imageDepth);
								#ifdef DEBUG_HIGH
								printf("Value in input: %0.2f, value in kernel %0.2f, value in output %0.2f\n\n", access(in, kx, ky, kz, imageWidth, imageHeight, imageDepth), weight, out[outIdx]);
								#endif
							}
						}
					}
				}
			}
		}
	}
}

void getKernelError(int errorWidth, int errorHeight, int kernelWidth, int kernelHeight, double **kernel, double *error, double **kernelError) {
	for (int x = 0; x < kernelWidth; x++) {
		for (int y = 0; y < kernelHeight; y++) {
			kernelError[x][y] = 0;
		}
	}
	for (int x = 0; x < errorWidth - kernelWidth; x++) {
		for (int y = 0; y < errorHeight - kernelHeight; y++) {
			for (int kx = x; kx < x + kernelWidth; kx++) {
				for (int ky = y; ky < y + kernelHeight; ky++) {
					kernelError[kx - x][ky - y] += kernel[kx - x][ky - y] * error[kx * errorWidth + ky];
				}
			}
		}
	}
}
#define POOL_META_SIZE 7
#define POOL_KERNEL_W 4
#define POOL_KERNEL_H 5
#define POOL_KERNEL_D 6
#define POOL_TYPE 'P'
Layer *create_pooling_layer(int kernelWidth, int kernelHeight, int kernelDepth) {
	if (!empty_layer_initialized) {
		empty_layer_initialized = 1;
		initializeNN();
	}
	Layer *out = malloc(sizeof(Layer));
	if (!out) {
		perror("malloc Layer");
		exit(1);
	}
	// Set this in 
	/*int w = width / out->metaparams[POOL_KERNEL_W];
	int h = height / out->metaparams[POOL_KERNEL_H];
	int d = depth / out->metaparams[POOL_KERNEL_D];
	out->metaparams[LAYER_X_DIMENSIONS_IDX] = w;
	out->metaparams[LAYER_Y_DIMENSIONS_IDX] = h;
	out->metaparams[LAYER_Z_DIMENSIONS_IDX] = d;
	*/
	
	out->output = EMPTY_LAYER;
	out->input = EMPTY_LAYER;
	out->metaparams = malloc(POOL_META_SIZE * sizeof(int));
	out->metaparams[LAYER_DIMENSIONS_IDX] = 3;
	out->metaparams[POOL_KERNEL_W] = kernelWidth;
	out->metaparams[POOL_KERNEL_H] = kernelHeight;
	out->metaparams[POOL_KERNEL_D] = kernelDepth;
	// Allocate memory for each part of the layer
	//out->values = malloc(size * sizeof(double));
	//out->error = malloc(size * sizeof(double));
	out->type = POOL_TYPE;
	
	out->allocated = 0;
	
	out->activate = activate_pool_layer;
	out->back_activate_error = back_activate_error_pool_layer;
	out->update_weights = update_weights_pool_layer;

	// This is a one dimensional array with all of the kernels values
	//out->weights = malloc(kernels * kernelWidth * kernelHeight * depth * sizeof(double));
	return out;
}

void activate_pool_layer(Layer *layer) {
	// Loop each pixel in the output layer
	for (int x = 0; x < layer->output->metaparams[LAYER_X_DIMENSIONS_IDX]; x++) {
		for (int y = 0; y < layer->output->metaparams[LAYER_Y_DIMENSIONS_IDX]; y++) {
			for (int z = 0; z < layer->output->metaparams[LAYER_Z_DIMENSIONS_IDX]; z++) {
				// Find the max in this layers values
				double max = -100000.0;
				for (int px = x * layer->metaparams[POOL_KERNEL_W]; x++; x < (x + 1) * layer->metaparams[POOL_KERNEL_W]) {
					for (int py = y * layer->metaparams[POOL_KERNEL_H]; y++; y < (y + 1) * layer->metaparams[POOL_KERNEL_H]) {
						for (int pz = z * layer->metaparams[POOL_KERNEL_D]; z++; z < (z + 1) * layer->metaparams[POOL_KERNEL_D]) {
							double value = layer->values[pz * layer->metaparams[LAYER_X_DIMENSIONS_IDX] * layer->metaparams[LAYER_Y_DIMENSIONS_IDX] + px * layer->metaparams[LAYER_Z_DIMENSIONS_IDX] + py];
							// TODO: Remember the location of each max for back propagation
							if (value > max) {
								max = value;
							}
						}
					}
				}
				layer->output->values[z * layer->output->metaparams[LAYER_X_DIMENSIONS_IDX] * layer->output->metaparams[LAYER_Y_DIMENSIONS_IDX] + x * layer->output->metaparams[LAYER_Z_DIMENSIONS_IDX] + y] = max;
			}
		}
	}
}
void back_activate_error_pool_layer(Layer *layer) {
	// TODO: Bring error back based on which neurons were max
}
void update_weights_pool_layer(Layer *outputLayer, double learning_rate) {
	// Do nothing because pool layers don't have weights
}

#define RESHAPE_META_SIZE 4
#define RESHAPE_TYPE 'R'
Layer *create_reshape_layer(int w, int h, int d, int dim) {
	if (!empty_layer_initialized) {
		empty_layer_initialized = 1;
		initializeNN();
	}
	Layer *out = malloc(sizeof(Layer));
	if (!out) {
		perror("malloc Layer");
		exit(1);
	}
	
	int size = w * h * d;
	out->output = EMPTY_LAYER;
	out->input = EMPTY_LAYER;
	out->metaparams = malloc(RESHAPE_META_SIZE * sizeof(int));
	out->metaparams[LAYER_DIMENSIONS_IDX] = dim;
	out->metaparams[LAYER_X_DIMENSIONS_IDX] = w;
	out->metaparams[LAYER_Y_DIMENSIONS_IDX] = h;
	out->metaparams[LAYER_Z_DIMENSIONS_IDX] = d;
	// Allocate memory for each part of the layer
	out->values = malloc(size * sizeof(double));
	out->error = malloc(size * sizeof(double));
	out->type = RESHAPE_TYPE;
	
	out->activate = activate_reshape_layer;
	out->back_activate_error = back_activate_error_reshape_layer;
	out->update_weights = update_weights_pool_layer;
	
	

	out->allocated = 0;

	// This is a one dimensional array with all of the kernels values
	//out->weights = malloc(kernels * kernelWidth * kernelHeight * depth * sizeof(double));
	if (!(out->values && out->error)) {
		perror("Failed to allocate memory");
		exit(1);
	}
	for (int i = 0; i < size; i++) {
		out->values[i] = 0;
		out->error[i] = 0;
	}
	return out;
}

void activate_reshape_layer(Layer *layer) {
	//printf("Reshaping from %d->%d/%c\n", layer->metaparams[LAYER_DIMENSIONS_IDX], layer->output->metaparams[LAYER_DIMENSIONS_IDX], layer->output->type);
	for (int i = 0; i < layer->metaparams[LAYER_X_DIMENSIONS_IDX] * layer->metaparams[LAYER_Y_DIMENSIONS_IDX] * layer->metaparams[LAYER_Z_DIMENSIONS_IDX]; i++) {
		layer->output->values[i] = layer->values[i];
	}
}
void back_activate_error_reshape_layer(Layer *layer) {
	for (int i = 0; i < layer->metaparams[LAYER_X_DIMENSIONS_IDX] * layer->metaparams[LAYER_Y_DIMENSIONS_IDX] * layer->metaparams[LAYER_Z_DIMENSIONS_IDX]; i++) {
		layer->error[i] = layer->output->error[i];
	}
}
void update_weights_reshape_layer(Layer *outputLayer, double learning_rate) {
	// Do nothing because reshape layers don't have weights
}

void connect_layers(Layer *input, Layer *output) {
	int inSize = input->metaparams[LAYER_DIMENSIONS_IDX];
	int outSize = output->metaparams[LAYER_DIMENSIONS_IDX];
	
	int w = input->metaparams[LAYER_X_DIMENSIONS_IDX];
	int h = input->metaparams[LAYER_Y_DIMENSIONS_IDX];
	int d = input->metaparams[LAYER_Z_DIMENSIONS_IDX];
	if (input->type == CONV_TYPE) {
		w /= input->metaparams[CONV_STRIDE_X];
		h /= input->metaparams[CONV_STRIDE_Y];
		d = input->metaparams[CONV_KERNEL_COUNT];
	}
	else if (input->type == POOL_TYPE) {
		w /= input->metaparams[POOL_KERNEL_W];
		h /= input->metaparams[POOL_KERNEL_H];
		d /= input->metaparams[POOL_KERNEL_D];
	}
	
	#ifdef DEBUG
	printf("Input: %d %c, output: %d %c\n", inSize, input->type, outSize, output->type);
	#endif
	
	// TODO: fix automatic redimensionizer
	if (inSize != outSize && input->type != RESHAPE_TYPE) {
		if (outSize != 1) {
			printf("Cannot convert %dd to %dd. Add a reshape layer in between\n", inSize, outSize);
		} // Reshape isn't actually needed to go to 1 because all values are encoded as 1 dimensional. I don't know why I did this
		else {
			printf("Attaching reshape with %d,%d,%d by %d to connect a %c(%d,%d,%d)->%c(%d,%d,%d)\n", w, h, d, inSize, input->type, input->metaparams[LAYER_X_DIMENSIONS_IDX], input->metaparams[LAYER_Y_DIMENSIONS_IDX], input->metaparams[LAYER_Z_DIMENSIONS_IDX], output->type, output->metaparams[LAYER_X_DIMENSIONS_IDX], output->metaparams[LAYER_Y_DIMENSIONS_IDX], output->metaparams[LAYER_Z_DIMENSIONS_IDX]);
			//*/// Attach a reshape layer
			Layer *reshape = create_reshape_layer(w, h, d, inSize);
			connect_layers(input, reshape);
			connect_layers(reshape, output);
		}
	}
	else {
		if (input->allocated) { 
			free(input->weights);
		
			input->allocated = 0;
		}
		input->output = output;
		output->input = input;
		if (input->type == CONV_TYPE) {
			#ifdef DEBUG
			printf("Allocating weights for a convolutional layer with %d kernels at %d,%d,%d\n", input->metaparams[CONV_KERNEL_COUNT], input->metaparams[CONV_KERNEL_W], input->metaparams[CONV_KERNEL_H], input->metaparams[CONV_KERNEL_D]);
			#endif
			int size = input->metaparams[CONV_KERNEL_COUNT] * input->metaparams[CONV_KERNEL_D] * input->metaparams[CONV_KERNEL_H] * input->metaparams[CONV_KERNEL_W];
			input->weights = malloc(size * sizeof(double));
			
			for (int i = 0; i < size; i++) {
				input->weights[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
			}
		}
		if (input->type == FF_TYPE) {
			input->allocated = 1;
			input->weights = malloc(input->metaparams[FF_NEURONS] * output->metaparams[FF_NEURONS] * sizeof(double));
			for (int i = 0; i < input->metaparams[FF_NEURONS] * output->metaparams[FF_NEURONS]; i++) {
				input->weights[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
			}
		}
		if (output->type == POOL_TYPE) {
			output->metaparams[LAYER_X_DIMENSIONS_IDX] = w;
			output->metaparams[LAYER_Y_DIMENSIONS_IDX] = h;
			output->metaparams[LAYER_Z_DIMENSIONS_IDX] = d;
			//TODO: finish allocation
		}
		
		// Allocate the weights for the input
		if (output->type == CONV_TYPE) {
			free(output->values);
			free(output->error);
			free(output->bias);
			printf("Generated sizes for convolutional network: %d,%d,%d\n", w,h,d);
			output->metaparams[LAYER_X_DIMENSIONS_IDX] = w;
			output->metaparams[LAYER_Y_DIMENSIONS_IDX] = h;
			output->metaparams[LAYER_Z_DIMENSIONS_IDX] = d;
			output->metaparams[CONV_KERNEL_D] = output->metaparams[LAYER_Z_DIMENSIONS_IDX];
			
			int outputSize = output->metaparams[LAYER_X_DIMENSIONS_IDX] * output->metaparams[LAYER_Y_DIMENSIONS_IDX] * output->metaparams[LAYER_Z_DIMENSIONS_IDX];
			output->values = malloc(outputSize * sizeof(double));
			output->error = malloc(outputSize * sizeof(double));
			output->bias = malloc(outputSize * sizeof(double));
			for (int i = 0; i < outputSize; i++) {
				output->values[i] = 0;
				output->error[i] = 0;
				output->bias[i] = (double) rand() / RAND_MAX * 2.0 - 1.0;
			}
			output->allocated = 1;
			input->allocated = 1;
			
			if (!(output->values && output->bias && output->error)) {
				perror("Failed to allocate memory");
				exit(1);
			}
		}
	}
}

void summarizeNetwork(Layer *layer) {
	printf("------------------------\n");
	while (layer != EMPTY_LAYER) {
		printLayerInfo(layer);
		layer = layer->output;
	}
	printf("------------------------\n");
}

double find_error(Layer *firstLayer, double **x, double **y, int size, int xlen, int ylen) {
	Layer *lastLayer = find_last_layer(firstLayer);
	double error = 0;
	for (int i = 0; i < size; i++) {
		double mse = 0;
		set_layer_values(firstLayer, x[i]);
		forward_propagate_layer(firstLayer);
		
		for (int j = 0; j < ylen; j++) {
			mse += (lastLayer->values[j] - y[i][j]) * (lastLayer->values[j] - y[i][j]);
			//printf("%f -> %f\n", lastLayer->values[j], y[i][j]);
		}

		mse /= ylen;
		error += mse;
	}
	return error / size;
}