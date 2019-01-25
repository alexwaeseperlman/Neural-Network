#include "util.h"
#include <stdio.h>

double sigmoid(double n) {
	return 1 / (1 + exp(-n));
}

// Takes a value that already has had sigmoid applied
double sigmoid_prime(double n) {
	return n * (1 - n);
}

double linear(double n) {
	return n;
}
double linear_prime(double n) {
	return 1;
}

#define leakyReluConstant (0.1)
double relu(double n) {
	return n > 0 ? n : leakyReluConstant * n;
}

double relu_prime(double n) {
	return n > 0 ? 1 : leakyReluConstant;
}