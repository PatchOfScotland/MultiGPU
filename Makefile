#Aarhus
FLAGS=-Xcompiler -fopenmp -O3 -arch=native -lineinfo
#Hendirx
#FLAGS=-Xcompiler -fopenmp -O3
PROGRAMS=map reduce matmul matmul_sm cannon_dev
BUILD_DIR=default

all: $(PROGRAMS)

map: 
	mkdir -p build/$(BUILD_DIR)
	nvcc src/map.cu -o $^ build/$(BUILD_DIR)/map $(FLAGS)

reduce: 
	mkdir -p build/$(BUILD_DIR)
	nvcc src/reduce.cu -o $^ build/$(BUILD_DIR)/reduce $(FLAGS)

matmul:
	mkdir -p build/$(BUILD_DIR)
	nvcc src/matmul.cu -o $^ build/$(BUILD_DIR)/matmul $(FLAGS)

matmul_sm:
	mkdir -p build/$(BUILD_DIR)
	nvcc src/matmul_sm.cu -o $^ build/$(BUILD_DIR)/matmul_sm $(FLAGS)

cannon_dev:
	mkdir -p build/$(BUILD_DIR)
	nvcc src/cannon_dev.cu -o $^ build/$(BUILD_DIR)/cannon_dev $(FLAGS)

hendrix:
	mkdir -p build/$(BUILD_DIR)
	nvcc src/map.cu -o $^ build/$(BUILD_DIR)/map $(FLAGS)
	nvcc src/reduce.cu -o $^ build/$(BUILD_DIR)/reduce $(FLAGS)
	nvcc src/matmul.cu -o $^ build/$(BUILD_DIR)/matmul $(FLAGS)

map_bench:
	make map
	./build/$(BUILD_DIR)/map 1000000000 100 -v

reduce_bench:
	make reduce
	./build/$(BUILD_DIR)/reduce 1000000000 100 -v

clean: 
	rm -f build/$(BUILD_DIR)/*

sanity_check:
	mkdir -p build/$(BUILD_DIR)
	nvcc src/sanity_check.cu -o $^ build/$(BUILD_DIR)/sanity_check $(FLAGS)
