INC = -I. -I/usr/local/cuda/samples/common/inc
NVCC_FLAGS := -arch=sm_50 --cudart shared -g --ptxas-options=-v

all: va.o mm.o sd.o fasthog.a convbench.a va_cpu.o mm_cpu.o

va_cpu.o: CPU/va.c
	gcc -c CPU/va.c -o va_cpu.o

mm_cpu.o: CPU/mm.c
	gcc -c CPU/mm.c -o mm_cpu.o

va.o: GPU/va.cu
	nvcc -c GPU/va.cu -o va.o $(NVCC_FLAGS)

mm.o: GPU/mm.cu
	nvcc -c GPU/mm.cu -o mm.o $(NVCC_FLAGS)

sd.o: GPU/StereoDisparity/sd.cu
	nvcc -c GPU/StereoDisparity/sd.cu -o sd.o $(NVCC_FLAGS) $(INC)

fasthog.a:
	cd GPU/FastHOG && make fasthog.a
	cp GPU/FastHOG/fasthog.a fasthog.a

convbench.a:
	cd GPU/convbench && make convbench.a
	cp GPU/convbench/convbench.a convbench.a

clean:
	cd GPU/FastHOG && make clean
	cd GPU/convbench && make clean
	rm -f *.o fasthog.a convbench.a
