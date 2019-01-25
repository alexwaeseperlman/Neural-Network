#define verbose 0
#define forward_prop_debug 0
typedef struct Layer {
	int neurons;
	double *values;
	double *bias;
	int output_neurons;
	double **weights;
	double (*activation)(double);
	double (*activation_prime)(double);
} Layer;

// Create a layer
Layer *create_layer(int neurons, int outputNeurons, double (*activation)(double), double (*activation_prime)(double));

// Set the values of a layer
void set_layer_values(Layer *layer, double values[]);

// Forward propagate a list of layers to other layers
void forward_propagate(Layer **layers, int number_of_layers);
void reverse_activate(Layer **layers, int number_of_layers);
void reset_values(Layer **layers, int number_of_layers);

// Back propagate a layer
// Returns target values for a hidden layer
double *back_propagate(Layer *inputLayer, Layer *outputLayer, double *error, double learningRate);
void train_layers(Layer **layers, int number_of_layers, double learningRate, double targetOutputs[]);
void train_on_dataset(Layer **layers, int number_of_layers, double **x, double **y, int number_of_records, int epochs,
                      double learning_rate);

void print_layer_values(Layer *inputLayer);
void print_network(Layer **inputLayers, int number_of_layers);

typedef struct Network {
	int number_of_layers;
	Layer **layers;
} Network;

Network *create_network(int, ...);
double *predict(Network *network, double, ...);
