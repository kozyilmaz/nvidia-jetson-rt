SAMPLE_DIR = ../Samples
INC = -I$(SAMPLE_DIR)
HOSTARCH = $(shell uname -m)
NVCC_FLAGS := -arch=sm_50 --cudart shared -g --ptxas-options=-v

all: benchmark_va benchmark_mm benchmark_sd benchmark_fasthog \
	benchmark_va_cpu benchmark_mm_cpu benchmark_convbench

benchmark_va: benchmark.o
	cd $(SAMPLE_DIR) && make va.o
	nvcc benchmark.o $(SAMPLE_DIR)/va.o -o benchmark_va $(NVCC_FLAGS)

benchmark_va_cpu: benchmark.o
	cd $(SAMPLE_DIR) && make va_cpu.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_cpu.o -o benchmark_va_cpu $(NVCC_FLAGS)

benchmark_mm: benchmark.o
	cd $(SAMPLE_DIR) && make mm.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm.o -o benchmark_mm $(NVCC_FLAGS)

benchmark_mm_cpu: benchmark.o
	cd $(SAMPLE_DIR) && make mm_cpu.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_cpu.o -o benchmark_mm_cpu $(NVCC_FLAGS) -lm

benchmark_sd: benchmark.o
	cd $(SAMPLE_DIR) && make sd.o
	nvcc benchmark.o $(SAMPLE_DIR)/sd.o -o benchmark_sd $(NVCC_FLAGS)

benchmark_fasthog: benchmark.o
	cd $(SAMPLE_DIR) && make fasthog.a
	nvcc benchmark.o $(SAMPLE_DIR)/fasthog.a $(SAMPLE_DIR)/GPU/FastHOG/libs/libfreeimage_$(HOSTARCH).a \
		$(NVCC_FLAGS) -o benchmark_fasthog

benchmark_convbench: benchmark.o
	cd $(SAMPLE_DIR) && make convbench.a
	nvcc benchmark.o $(SAMPLE_DIR)/convbench.a $(NVCC_FLAGS) -o benchmark_convbench

benchmark.o: ./benchmark.c
	gcc -c ./benchmark.c -Wall -g $(INC)

clean:
	rm -f *.o benchmark_*
	cd $(SAMPLE_DIR) && make clean
