
all: mixbench_all rodinia_all

clean: mixbench_clean rodinia_clean

mixbench_all:
	make -C mixbench mixbench-cuda-alt mixbench-cuda-ro

mixbench_clean:
	make -C mixbench clean

rodinia_all:
	if [ ! -f "rodinia/.patched" ]; then \
		patch -d rodinia -p1 < patches/gpu-rodinia.patch; \
		touch rodinia/.patched; \
	fi
	mkdir -p $(CURDIR)/bin/linux/cuda
	make -C rodinia CUDA

rodinia_clean:
	make -C rodinia CUDA_clean
