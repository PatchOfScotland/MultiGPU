FLAGS=-Xcompiler -fopenmp -O3
PROGRAMS=map reduce tiled_scan

all: $(PROGRAMS)

map: 
	mkdir -p build
	nvcc src/map.cu -o $^ build/map $(FLAGS)

reduce: 
	mkdir -p build
	nvcc src/reduce.cu -o $^ build/reduce $(FLAGS)

tiled_scan:
	mkdir -p build
	g++ src/tiled_scan.cpp -o $^ build/tiled_scan -fopenmp -O3

map_bench:
	make map
	./build/map 1000000000 100 -v

reduce_bench:
	make reduce
	./build/reduce 1000000000 100 -v

clean: 
	rm -f build/*