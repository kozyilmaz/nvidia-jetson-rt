
all: caffe_all periodictaskreleaser_all cudamemoryexperiments_all mixbench_all rodinia_all

clean: caffe_clean periodictaskreleaser_clean cudamemoryexperiments_clean mixbench_clean rodinia_clean


# caffe
caffe_all:
	if [ ! -f "caffe/.patched" ]; then \
		patch -d caffe -p1 < patches/caffe.patch; \
		touch caffe/.patched; \
	fi
	cp caffe/Makefile.config.example caffe/Makefile.config
	make -C caffe -j4 all
	make -C caffe -j4 test
	make -C caffe runtest

caffe_clean:
	make -C caffe clean


# nheqminer
nheqminer_all:
	( mkdir -p nheqminer-gpu; cd nheqminer-gpu; \
		cmake -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib" \
		-DUSE_CPU_TROMP=OFF -DUSE_CUDA_TROMP=OFF -DUSE_CPU_XENONCAT=OFF -DUSE_CUDA_DJEZO=ON ../nheqminer; \
	)

nheqminer_clean:
	rm -rf nheqminer-gpu



# periodictaskreleaser
periodictaskreleaser_all:
	if [ ! -f "periodictaskreleaser/.patched" ]; then \
		patch -d periodictaskreleaser -p1 < patches/periodictaskreleaser.patch; \
		touch periodictaskreleaser/.patched; \
	fi
	make -C periodictaskreleaser/Benchmark -j4

periodictaskreleaser_clean:
	make -C periodictaskreleaser/Benchmark clean


# cudamemoryexperiments
cudamemoryexperiments_all:
	if [ ! -f "cudamemoryexperiments/.patched" ]; then \
		patch -d cudamemoryexperiments -p1 < patches/cudamemoryexperiments.patch; \
		touch cudamemoryexperiments/.patched; \
	fi
	make -C cudamemoryexperiments/InOrderWalkExperiments -j4
	make -C cudamemoryexperiments/MemCpyExperiments -j4
	make -C cudamemoryexperiments/RandomWalkExperiments -j4

cudamemoryexperiments_clean:
	make -C cudamemoryexperiments/InOrderWalkExperiments clean
	make -C cudamemoryexperiments/MemCpyExperiments clean
	make -C cudamemoryexperiments/RandomWalkExperiments clean
	rm -rf cudamemoryexperiments/*.txt


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

