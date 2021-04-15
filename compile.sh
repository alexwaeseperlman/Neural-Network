rm bin/program
for i in `ls *.c`; do
	gcc -c -g -fopenmp $i -O3 -o object/$i.o -lm --std=c11 -Wall -pedantic -pthread -Wno-traditional
done
gcc -o bin/program object/*.o -fopenmp -O3 -lm --std=c11 -Wall -pedantic -pthread -Wno-traditional
