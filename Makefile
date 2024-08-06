#Aarhus
#FLAGS=-Xcompiler -fopenmp -O3 -arch=native
#Hendirx
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

memtest:
	mkdir -p build
	nvcc src/memtest.cu -o $^ build/memtest $(FLAGS)

matmul_sm:
	mkdir -p build
	nvcc src/matmul_sm.cu -o $^ build/matmul_sm $(FLAGS)

cannon_dev:
	mkdir -p build
	nvcc src/cannon_dev.cu -o $^ build/cannon_dev $(FLAGS)

hendrix:
	mkdir -p build
	nvcc src/map.cu -o $^ build/map $(FLAGS)
	nvcc src/reduce.cu -o $^ build/reduce $(FLAGS)
	nvcc src/matmul.cu -o $^ build/matmul $(FLAGS)

map_bench:
	make map
	./build/map 1000000000 100 -v

reduce_bench:
	make reduce
	./build/reduce 1000000000 100 -v

clean: 
	rm -f build/*

sanity_check:
	mkdir -p build
	nvcc src/sanity_check.cu -o $^ build/sanity_check $(FLAGS)