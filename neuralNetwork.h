#define verbose 0
typedef struct Layer
{
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
void reset_values(Layer **layers, int number_of_layers);

// Back propagate a layer
// Returns target values for a hidden layer
double *
back_propagate(Layer *inputLayer, Layer *outputLayer, double *error, double learningRate);
void train_network(Layer **layers, int number_of_layers, double learningRate, double targetOutputs[]);

void print_layer_values(Layer *inputLayer);
