.PHONY: all clean
NVCC_FLAGS := -arch=sm_53 --cudart shared -g --ptxas-options=-v

all: random_walk

random_walk: random_walk.cu
	nvcc -o random_walk random_walk.cu $(NVCC_FLAGS)

clean:
	rm -f random_walk
