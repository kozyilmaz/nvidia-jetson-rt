# Performance Benchmarks

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
$ git push
# check caffe subtree
$ git fetch https://github.com/BVLC/caffe master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/caffe
# sync caffe subtree
$ git remote add caffe-remote https://github.com/BVLC/caffe
$ git subtree pull --prefix=benchmarks/caffe/ --squash caffe-remote master
```

#### AlexNet
Benchmark [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) described in Google's [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) paper
```shell
$ caffe/build/tools/caffe time --model=caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 200
```

#### GoogleNet
Benchmark [GoogleNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) described in Google's [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) paper
```shell
$ caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 200
```


## [Nheqminer](https://github.com/nicehash/nheqminer)
Equihash miner for NiceHash
```shell
# add nheqminer as subtree
$ git remote add nheqminer https://github.com/nicehash/nheqminer.git
$ git subtree add --prefix=benchmarks/nheqminer/ --squash nheqminer master
$ git push
# check nheqminer subtree
$ git fetch https://github.com/nicehash/nheqminer.git master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/nheqminer
# sync nheqminer subtree
$ git remote add nheqminer-remote https://github.com/nicehash/nheqminer.git
$ git subtree pull --prefix=benchmarks/nheqminer/ --squash nheqminer-remote master
```


## [Periodic Task Releaser](https://github.com/yalue/PeriodicTaskReleaser)
This is a collection of CUDA programs intended to measure interference between GPU processes. It was created as part of ongoing research in real-time systems (see [paper](http://cs.unc.edu/~anderson/papers/ospert16.pdf)) at UNC Chapel Hill
```shell
# add periodictaskreleaser as subtree
$ git remote add periodictaskreleaser https://github.com/yalue/PeriodicTaskReleaser.git
$ git subtree add --prefix=benchmarks/periodictaskreleaser/ --squash periodictaskreleaser master
$ git push
# check periodictaskreleaser subtree
$ git fetch https://github.com/yalue/PeriodicTaskReleaser.git master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/periodictaskreleaser
# sync periodictaskreleaser subtree
$ git remote add periodictaskreleaser-remote https://github.com/yalue/PeriodicTaskReleaser.git
$ git subtree pull --prefix=benchmarks/periodictaskreleaser/ --squash periodictaskreleaser-remote master
```

Commands below create graphs from `periodictaskreleaser` benchmarks
```shell
# install prerequisites
$ sudo apt install libfreetype6-dev
$ sudo apt install python-pip
$ sudo apt install python-gi-cairo
$ pip install matplotlib

# create performance test graphs
$ python generate_plots.py
```


## [CUDA Memory Experiments](https://github.com/Sarahild/CudaMemoryExperiments)
Simple programs that run memory experiments on CUDA, created at UNC - Chapel Hill
```shell
# add cudamemoryexperiments as subtree
$ git remote add cudamemoryexperiments https://github.com/Sarahild/CudaMemoryExperiments.git
$ git subtree add --prefix=benchmarks/cudamemoryexperiments/ --squash cudamemoryexperiments master
$ git push
# check cudamemoryexperiments subtree
$ git fetch https://github.com/Sarahild/CudaMemoryExperiments.git master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/cudamemoryexperiments
# sync cudamemoryexperiments subtree
$ git remote add cudamemoryexperiments-remote https://github.com/Sarahild/CudaMemoryExperiments.git
$ git subtree pull --prefix=benchmarks/cudamemoryexperiments/ --squash cudamemoryexperiments-remote master
```


## [Mixbench](https://github.com/ekondis/mixbench)
A GPU benchmark tool for evaluating GPUs on mixed operational intensity kernels (CUDA, OpenCL, HIP)
```shell
# add mixbench as subtree
$ git remote add mixbench https://github.com/ekondis/mixbench.git
$ git subtree add --prefix=benchmarks/mixbench/ --squash mixbench master
$ git push
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
$ git push
# check rodinia subtree
$ git fetch https://github.com/yuhc/gpu-rodinia.git master
$ ./contrib/devtools/git-subtree-check.sh benchmarks/rodinia
# sync rodinia subtree
$ git remote add rodinia-remote https://github.com/yuhc/gpu-rodinia.git
$ git subtree pull --prefix=benchmarks/rodinia/ --squash rodinia-remote master
```

Please check [nvpmodel](http://www.jetsonhacks.com/2017/03/24/caffe-deep-learning-framework-nvidia-jetson-tx2/) and explore Max-Q, Max-P and Max-N power modes.
