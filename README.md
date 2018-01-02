# Real-Time Scheduling with NVIDIA Jetson TX2

## NVIDIA Jetson TX2 Configuration
* Visit [Nvidia L4T download directory](https://developer.nvidia.com/embedded/linux-tegra) for the list of packages  
Latest stable release at the time for Jetson TX2 platform is "L4T 28.1 - Production Version"  
Download all packages to `$HOME/nvidia`, rest of the document assumes that
```
1. NVIDIA_Tegra_Linux_Driver_Package.tar : Documentation
2. Tegra186_Linux_R28.1.0_aarch64.tbz2 : Jetson TX2 64-bit Driver Package
3. Tegra_Linux_Sample-Root-Filesystem_R28.1.0_aarch64.tbz2 : Sample Root File System
4. gcc-4.8.5-aarch64.tgz : GCC 4.8.5 Tool Chain for 64-bit BSP (which contains gcc-4.8.5 Jetson TX2 release for cross-compilation)
5. source_release.tbz2 Source Packages : (which contains Linux kernel sources for Jetson TX2 platform)

Optionally you may download JetPack to extract CUDA .deb packages or download them directly
6. JetPack-L4T-3.1-linux-x64.run: All-in-one package Jetson SDK containing CUDA, TensorRT, cuDNN, VisionWorks/OpenCV4Tegra, Samples/Documentation
```

* [How to build an image and flash into eMMC](docs/README.00-flashing.md)
* [How to configure a fresh install](docs/README.01-configure.md)
* [How to develop and update board software](docs/README.02-development.md)

## Real-Time Patches and L4T Kernel
* [Real-Time Linux Project](https://rt.wiki.kernel.org/index.php/Main_Page) contains necessary information to add real-time capabilities to vanilla kernels  
* [How to patch and build a real-time Jetson kernel](docs/README.03-realtime.md)

## Performance Comparison (RT & non-RT)
Below are a selection of popular benchmark suites selected to probe different aspects of CPU/GPU hybrid systems
* [Caffe](https://github.com/BVLC/caffe) is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR)/The Berkeley Vision and Learning Center (BVLC) and community contributors
* [Mixbench](https://github.com/ekondis/mixbench) is a GPU benchmark tool for evaluating GPUs on mixed operational intensity kernels (CUDA, OpenCL, HIP)
* [Rodinia Benchmark Suite](https://github.com/yuhc/gpu-rodinia.git) is a collection of parallel programs which targets heterogeneous computing platforms with both multicore CPUs and GPUs

[How to build and run benchmark suites](docs/README.04-benchmarks.md)

