FLAGS=-Xcompiler -fopenmp -O3
PROGRAMS=map reduce matmul

all: $(PROGRAMS)

map: 
	mkdir -p build
	nvcc src/map.cu -o $^ build/map $(FLAGS)

reduce: 
	mkdir -p build
	nvcc src/reduce.cu -o $^ build/reduce $(FLAGS)

matmul:
	mkdir -p build
	nvcc src/matmul.cu -o $^ build/matmul $(FLAGS)

tiled_scan:
	mkdir -p build
	nvcc src/tiled_scan.cu -o $^ build/tiled_scan $(FLAGS)

map_bench:
	make map
	./build/map 1000000000 100 -v

reduce_bench:
	make reduce
	./build/reduce 1000000000 100 -v

clean: 
	rm -f build/*