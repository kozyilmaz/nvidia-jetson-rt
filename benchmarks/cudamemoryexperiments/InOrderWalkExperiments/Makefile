.PHONY: all clean
NVCC_FLAGS := -arch=sm_53 --cudart shared -g --ptxas-options=-v

all: inorder_walk

inorder_walk: inorder_walk.cu
	nvcc -o inorder_walk inorder_walk.cu $(NVCC_FLAGS)

clean:
	rm -f inorder_walk
