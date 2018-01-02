.PHONY: all clean
NVCC_FLAGS := -arch=sm_53 --cudart shared -g --ptxas-options=-v

all: memcpy

memcpy: memcpy.cu
	nvcc -o memcpy memcpy.cu $(NVCC_FLAGS)

clean:
	rm -f memcpy
