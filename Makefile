FLAGS=-O3
PROGRAMS=map

all: $(PROGRAMS)

map:
	mkdir -p build
	nvcc src/map.cu -o build/map $(FLAGS)

clean:
	rm -f build/$(PROGRAMS)