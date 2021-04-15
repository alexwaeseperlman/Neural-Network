#pragma once
#define verbose 0
#define forward_prop_debug 0

// All 3d arrays are encoded as depth > width > height
// Example usage: arr[z * width * height + x * height + y]
#define access3d(arr, x, y, z, width, height, depth) arr[z * width * height + x * height + y]

struct Layer;
typedef void (*method)(struct Layer* self);
typedef void (*update_weights)(struct Layer* self, double learning_rate);

typedef struct Layer {
	int *metaparams;
	double *values;
	double *error;
	double *bias;
	double *weights;
	double (*activation)(double);
	double (*activation_prime)(double);
	method activate;
	method back_activate_error;
	update_weights update_weights;
	struct Layer *output;
	struct Layer *input;
	char type;
	int allocated;
} Layer;


// Sets up all the globals that this library needs
void initializeNN();

// Create a layer
Layer *create_feedforward_layer(int neurons, double (*activation)(double), double (*activation_prime)(double));
Layer *create_convolution_layer(int strideX, int strideY, int kernelWidth, int kernelHeight, int kernels, double (*activation)(double), double (*activation_prime)(double));

// Connect two layers and initialize the weights randomly
void connect_layers(Layer *input, Layer *output);
void connect_all_layers(Layer **layers, int number_of_layers);

Layer *find_last_layer(Layer *input);

// Set the values of a layer
void set_layer_values(Layer *layer, double values[]);

// Reset the values of a layer
void reset_values_layer(Layer *layer);
void reset_values_recursive_layer(Layer *firstLayer);

void reset_errors_layer(Layer *layer);
void reset_errors_recursive_layer(Layer *firstLayer);

// Activate a layer by setting the values of its output
void activate_feedforward_layer(Layer *layer);
void activate_convolutional_layer(Layer *layer);
void activate_pool_layer(Layer *layer);
void activate_reshape_layer(Layer *layer);
// Activate layers recursively
void forward_propagate_layer(Layer *firstLayer);

// Backwards propagate the error and multiply by the derivative as it goes
void back_activate_error_feedforward_layer(Layer *layer);
void back_activate_error_convolution_layer(Layer *layer);
void back_activate_error_pool_layer(Layer *layer);
void back_activate_error_reshape_layer(Layer *layer);
void back_propagate_error_layer(Layer *lastLayer);

// Update the weights of a layer based on it's error
void update_weights_feedforward_layer(Layer *outputLayer, double learning_rate);
void update_weights_convolution_layer(Layer *outputLayer, double learning_rate);
void update_weights_pool_layer(Layer *outputLayer, double learning_rate);

// Back propagate a layer
// Returns target values for a hidden layer
double *back_propagate(Layer *inputLayer, Layer *outputLayer, double *error, double learningRate);
void train_layers(Layer *layers, double learning_rate, double targetOutputs[]);
void train_on_dataset(Layer *layers, double **x, double **y, int number_of_records, int epochs, double learning_rate);

void print_layer_values(Layer *inputLayer);
void print_network(Layer **inputLayers, int number_of_layers);

void applyKernel(int imageWidth, int imageHeight, int imageDepth, int kernelWidth, int kernelHeight, int kernelDepth, int kernels, int strideX, int strideY, double *kernel, double *in, double *out, int forwards);
void getKernelError(int errorWidth, int errorHeight, int kernelWidth, int kernelHeight, double **kernel, double *error, double **kernelError);

void summarizeNetwork(Layer *inputLayer);

Layer *create_pooling_layer(int poolWidth, int poolHeight, int poolDepth);
Layer *create_reshape_layer(int outW, int outH, int outD, int dim);

double find_error(Layer *firstLayer, double **x, double **y, int size, int xlen, int ylen);
