rm program
gcc-8 -c -g -fopenmp *.c -O2
gcc-8 -o program *.o -fopenmp -O2
