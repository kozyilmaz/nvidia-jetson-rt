# GPU & CPU Benchmarks

## [Mixbench](https://github.com/ekondis/mixbench)
A GPU benchmark tool for evaluating GPUs on mixed operational intensity kernels (CUDA, OpenCL, HIP)
```shell
# add mixbench as subtree
$ git remote add mixbench https://github.com/ekondis/mixbench.git
$ git subtree add --prefix=benchmarks/mixbench/ --squash mixbench master
# check mixbench subtree
$ git fetch https://github.com/ekondis/mixbench.git master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/mixbench
# sync mixbench subtree
$ git remote add mixbench-remote https://github.com/ekondis/mixbench.git
$ git subtree pull --prefix=benchmarks/mixbench/ --squash mixbench-remote master
```


## [Rodinia](https://github.com/yuhc/gpu-rodinia.git)
The University of Virginia Rodinia Benchmark Suite is a collection of parallel programs which targets heterogeneous computing platforms with both multicore CPUs and GPUs.
```shell
# install prerequisites
$ sudo apt install libglew-dev

# add rodinia as subtree
$ git remote add rodinia https://github.com/yuhc/gpu-rodinia.git
$ git subtree add --prefix=benchmarks/rodinia/ --squash rodinia master
# check rodinia subtree
$ git fetch https://github.com/yuhc/gpu-rodinia.git master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/rodinia
# sync rodinia subtree
$ git remote add rodinia-remote https://github.com/yuhc/gpu-rodinia.git
$ git subtree pull --prefix=benchmarks/rodinia/ --squash rodinia-remote master
```


## [Caffe](https://github.com/BVLC/caffe)
```shell
# install prerequisites
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
$ sudo apt-get install --no-install-recommends libboost-all-dev
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
# BLAS
$ sudo apt-get install libatlas-base-dev
$ git clone https://github.com/BVLC/caffe
$ cd caffe
$ cp Makefile.config.example Makefile.config
$ make all
```

https://github.com/yalue/PeriodicTaskReleaser
https://github.com/Sarahild/CudaMemoryExperiments
