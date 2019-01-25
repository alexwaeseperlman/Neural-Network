#include "util.h"

double sigmoid(double n) {
	return 1 / (1 + exp(-n));
}

double linear(double n) {
	return n;
}

// Takes a value that already has had sigmoid applied
double sigmoid_prime(double n) {
	return n * (1 - n);
}

double linear_prime(double n) {
	return 1;
}