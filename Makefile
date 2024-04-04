FLAGS=-Xcompiler -fopenmp -O3
PROGRAMS=map reduce

all: $(PROGRAMS)

map: 
	mkdir -p build
	nvcc src/map.cu -o $^ build/map $(FLAGS)

reduce: 
	mkdir -p build
	nvcc src/reduce.cu -o $^ build/reduce $(FLAGS)

map_bench:
	make map
	./build/map 1000000000 100 -v

reduce_bench:
	make reduce
	./build/reduce 1000000000 100 -v

clean: 
	rm -f build/*