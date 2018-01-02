
all: caffe_all mixbench_all rodinia_all

clean: caffe_clean mixbench_clean rodinia_clean


# caffe
caffe_all:
	if [ ! -f "caffe/.patched" ]; then \
		patch -d caffe -p1 < patches/caffe.patch; \
		touch caffe/.patched; \
	fi
	cp caffe/Makefile.config.example caffe/Makefile.config
	make -C caffe -j4 all
	make -C caffe -j4 test
	make -C runtest

caffe_clean:
	make -C caffe clean


# periodictaskreleaser
periodictaskreleaser_all:
	make -C periodictaskreleaser/Benchmark

periodictaskreleaser_all:
	make -C periodictaskreleaser/Benchmark clean


# mixbench
mixbench_all:
	make -C mixbench -j4 mixbench-cuda-alt mixbench-cuda-ro

mixbench_clean:
	make -C mixbench clean


# rodinia
rodinia_all:
	if [ ! -f "rodinia/.patched" ]; then \
		patch -d rodinia -p1 < patches/gpu-rodinia.patch; \
		touch rodinia/.patched; \
	fi
	mkdir -p $(CURDIR)/rodinia/bin/linux/cuda
	make -C rodinia -j4 CUDA

rodinia_clean:
	make -C rodinia CUDA_clean

