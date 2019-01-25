rm program
for i in `ls *.c`; do
	gcc-8 -c -g -fopenmp $i -O2 -o object/$i.o
done
gcc-8 -o bin/program object/*.o -fopenmp -O2
