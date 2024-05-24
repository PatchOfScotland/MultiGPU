FLAGS=-Xcompiler -fopenmp -O3 -arch=native
FLAGS_HENDRIX=-Xcompiler -fopenmp -O3
PROGRAMS=map reduce matmul

all: $(PROGRAMS)

map: 
	mkdir -p build
	nvcc src/map.cu -o $^ build/map $(FLAGS_HENDRIX)

reduce: 
	mkdir -p build
	nvcc src/reduce.cu -o $^ build/reduce $(FLAGS_HENDRIX)

matmul:
	mkdir -p build
	nvcc src/matmul.cu -o $^ build/matmul $(FLAGS_HENDRIX)

hendrix:
	mkdir -p build
	nvcc src/map.cu -o $^ build/map $(FLAGS_HENDRIX)
	nvcc src/reduce.cu -o $^ build/reduce $(FLAGS_HENDRIX)
	nvcc src/matmul.cu -o $^ build/matmul $(FLAGS_HENDRIX)

map_bench:
	make map
	./build/map 1000000000 100 -v

reduce_bench:
	make reduce
	./build/reduce 1000000000 100 -v

clean: 
	rm -f build/*