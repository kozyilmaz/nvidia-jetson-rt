# GPU & CPU Benchmarks

## [Caffe](https://github.com/BVLC/caffe)
Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR)/The Berkeley Vision and Learning Center (BVLC) and community contributors.
```shell
# install prerequisites
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
$ sudo apt-get install --no-install-recommends libboost-all-dev
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
# BLAS
$ sudo apt-get install libatlas-base-dev

# add caffe as subtree
$ git remote add caffe https://github.com/BVLC/caffe
$ git subtree add --prefix=benchmarks/caffe/ --squash caffe master
# check caffe subtree
$ git fetch https://github.com/BVLC/caffe master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/caffe
# sync caffe subtree
$ git remote add caffe-remote https://github.com/BVLC/caffe
$ git subtree pull --prefix=benchmarks/caffe/ --squash caffe-remote master
```

Benchmark [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) described in Google's [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) paper
```shell
$ caffe/build/tools/caffe time --model= caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 30
```
Benchmark [GoogleNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) described in Google's [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) paper
```shell
$ caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 30
```

https://github.com/yalue/PeriodicTaskReleaser
https://github.com/Sarahild/CudaMemoryExperiments


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

Please check [nvpmodel](http://www.jetsonhacks.com/2017/03/24/caffe-deep-learning-framework-nvidia-jetson-tx2/) and explore Max-Q, Max-P and Max-N power modes.
