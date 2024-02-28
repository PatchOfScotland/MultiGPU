FLAGS=-O3
PROGRAMS=map

all: $(PROGRAMS)

map: 
	mkdir -p build
	nvcc src/map.cu -o $^ build/map $(FLAGS)

map_bench:
	make map
	./build/map 1000000000 100 -v

clean:
	rm -f build/$(PROGRAMS)