#define verbose 0
#define forward_prop_debug 0
typedef struct Layer {
	int neurons;
	double *values;
	double *error;
	double *bias;
	double **weights;
	double (*activation)(double);
	double (*activation_prime)(double);
	struct Layer *output;
	struct Layer *input;
} Layer;

// Sets up all the globals that this library needs
void initializeNN();

// Create a layer
Layer *create_layer(int neurons, double (*activation)(double), double (*activation_prime)(double));

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
void activate_layer(Layer *layer);
// Activate layers recursively
void forward_propagate_layer(Layer *firstLayer);

// Backwards propagate the error and multiply by the derivative as it goes
void back_activate_error_layer(Layer *layer);
void back_propagate_error_layer(Layer *lastLayer);

// Update the weights of a layer based on it's error
void update_weights_layer(Layer *outputLayer, double learning_rate);

// Back propagate a layer
// Returns target values for a hidden layer
double *back_propagate(Layer *inputLayer, Layer *outputLayer, double *error, double learningRate);
void train_layers(Layer *layers, double learning_rate, double targetOutputs[]);
void train_on_dataset(Layer *layers, double **x, double **y, int number_of_records, int epochs, double learning_rate);

void print_layer_values(Layer *inputLayer);
void print_network(Layer **inputLayers, int number_of_layers);

typedef struct Network {
	int number_of_layers;
	Layer **layers;
} Network;

Network *create_network(int, ...);
double *predict(Network *network, double, ...);
