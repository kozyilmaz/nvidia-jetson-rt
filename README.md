# Real-Time Scheduling with NVIDIA Jetson TX2

## NVIDIA Jetson TX2 Configuration
Visit [Nvidia L4T download directory](https://developer.nvidia.com/embedded/linux-tegra) for the list of packages  
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

* [How to build an image and flash into eMMC](README.00-flashing.md)
* [How to configure a fresh install](README.01-configure.md)


## Build Linux for Nvidia Jetson TX2

#### Prepare build sources
```shell
$ mkdir -p $HOME/nvidia
$ cd $HOME/nvidia
$ tar -xvf gcc-4.8.5-aarch64.tgz
$ tar -xvf source_release.tbz2
$ tar -xvf sources/kernel_src-tx2.tbz2
```

#### Create an "environment" file for envvars
```shell
$ cat $HOME/nvidia/environment
#!/bin/sh
# toolchain flags
export BSPTOOLCHAIN=$HOME/nvidia/install/bin
export PATH=${BSPTOOLCHAIN}:${PATH}
# kernel flags
export ARCH=arm64
export CROSS_COMPILE=aarch64-unknown-linux-gnu-
export TEGRA_KERNEL_OUT=$HOME/nvidia/tegra-jetson-tx2-kernel
```

#### Export build variables and start compiling
```shell
$ cd $HOME/nvidia
$ source environment
$ mkdir -p $TEGRA_KERNEL_OUT
$ cd $HOME/nvidia/kernel/kernel-4.4
# create default config
$ make O=$TEGRA_KERNEL_OUT ARCH=$ARCH tegra18_defconfig
# create compressed kernel image
$ make -j4  O=$TEGRA_KERNEL_OUT ARCH=$ARCH zImage
# compile device tree
$ make O=$TEGRA_KERNEL_OUT ARCH=$ARCH dtbs
# compile and install kernel modules
$ make -j4 O=$TEGRA_KERNEL_OUT ARCH=$ARCH modules
$ make O=$TEGRA_KERNEL_OUT ARCH=$ARCH modules_install INSTALL_MOD_PATH=$TEGRA_KERNEL_OUT/modules
```

#### Copy binaries to new "Linux_for_Tegra" for deployment
```shell
$ mkdir -p $HOME/nvidia/Linux_for_Tegra/kernel
$ cp $TEGRA_KERNEL_OUT/arch/arm64/boot/Image $HOME/nvidia/Linux_for_Tegra/kernel
$ cp $TEGRA_KERNEL_OUT/arch/arm64/boot/zImage $HOME/nvidia/Linux_for_Tegra/kernel
$ mkdir -p $HOME/nvidia/Linux_for_Tegra/kernel/dtb
$ cp $TEGRA_KERNEL_OUT/arch/arm64/boot/dts/*.dtb $HOME/nvidia/Linux_for_Tegra/kernel/dtb
$ cd $TEGRA_KERNEL_OUT/modules
$ tar --owner root --group root -cjf kernel_supplements.tbz2 *
$ cp $TEGRA_KERNEL_OUT/modules/kernel_supplements.tbz2 $HOME/nvidia/Linux_for_Tegra/kernel/kernel_supplements.tbz2
$ cd $HOME/nvidia
$ tar --owner root --group root -cjf $HOME/nvidia/Linux_for_Tegra.tbz2 Linux_for_Tegra
$ scp $HOME/nvidia/Linux_for_Tegra.tbz2 nvidia@JETSON_IP_ADDRESS:/home/nvidia
```

