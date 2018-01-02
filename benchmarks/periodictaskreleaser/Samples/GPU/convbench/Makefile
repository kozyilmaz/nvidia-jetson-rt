.PHONY: all clean

NVCC := nvcc -ccbin g++
NVCCFLAGS := --cudart shared -g --ptxas-options=-v -arch=sm_50
INCLUDES := -I/usr/local/cuda/samples/common/inc -I.

all: convbench.a

convbench.o: convbench.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

matrixMul.o: matrixMul.cu matrixMul.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

im2col.o: im2col.cu im2col.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

convbench.a: im2col.o matrixMul.o convbench.o
	ar rcsv convbench.a *.o
	ranlib convbench.a

clean:
	rm -f *.o convbench.a
