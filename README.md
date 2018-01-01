# Real-Time Scheduling with NVIDIA Jetson TX2

## NVIDIA Jetson TX2 Configuration
Visit [Nvidia L4T download directory](https://developer.nvidia.com/embedded/linux-tegra) for the list of packages  
Latest stable release at the time for Jetson TX2 platform is "L4T 28.1 - Production Version"  
Download all packages to `$HOME/nvidia`
```
1. NVIDIA_Tegra_Linux_Driver_Package.tar : Documentation
2. Tegra186_Linux_R28.1.0_aarch64.tbz2 : Jetson TX2 64-bit Driver Package
3. Tegra_Linux_Sample-Root-Filesystem_R28.1.0_aarch64.tbz2 : Sample Root File System
4. gcc-4.8.5-aarch64.tgz : GCC 4.8.5 Tool Chain for 64-bit BSP (which contains gcc-4.8.5 Jetson TX2 release for cross-compilation)
5. source_release.tbz2 Source Packages : (which contains Linux kernel sources for Jetson TX2 platform)

Optionally you may download JetPack to extract CUDA .deb packages or download them directly
6. JetPack-L4T-3.1-linux-x64.run: All-in-one package Jetson SDK containing CUDA, TensorRT, cuDNN, VisionWorks/OpenCV4Tegra, Samples/Documentation
```

## Create Jetson TX2 image and flash it on eMMC
```
$ mkdir -p $HOME/nvidia/docs
$ tar -xvf NVIDIA_Tegra_Linux_Driver_Package.tar -C $HOME/nvidia/docs
$ sudo tar -xvf Tegra186_Linux_R28.1.0_aarch64.tbz2
$ sudo tar -xvf Tegra_Linux_Sample-Root-Filesystem_R28.1.0_aarch64.tbz2 -C $HOME/nvidia/Linux_for_Tegra/rootfs
$ cd $HOME/nvidia/Linux_for_Tegra
$ sudo ./apply_binaries.sh
# put the board in recovery mode (RECOVERY FORCE+RESET)
$ sudo ./flash.sh -t jetson-tx2 mmcblk0p1
```

## Configure Jetson TX2 board

#### [HOST] Check for available SSH connection to Jetson TX2
```
# to connect Jetson board
$ ssh nvidia@192.168.42.71
# to copy files to Jetson board
$ scp setup.txt nvidia@192.168.42.71:/home/nvidia
```

#### [HOST] Copy CUDA/TensorFlow packages to Jetson TX2
```
$ scp cuda-repo-l4t-8-0-local_8.0.84-1_arm64.deb nvidia@192.168.42.71:/home/nvidia
$ scp libcudnn6_6.0.21-1+cuda8.0_arm64.deb nvidia@192.168.42.71:/home/nvidia
$ scp libcudnn6-dev_6.0.21-1+cuda8.0_arm64.deb nvidia@192.168.42.71:/home/nvidia
$ scp libcudnn6-doc_6.0.21-1+cuda8.0_arm64.deb nvidia@192.168.42.71:/home/nvidia
$ scp nv-gie-repo-ubuntu1604-ga-cuda8.0-trt2.1-20170614_1-1_arm64.deb nvidia@192.168.42.71:/home/nvidia
```

#### [TARGET] Update Ubuntu packages
```
$ sudo apt update
$ sudo apt upgrade
$ sudo apt autoremove
```

#### [TARGET] Install CUDA and TensorFlow
```
$ sudo dpkg -i /home/nvidia/cuda-repo-l4t-8-0-local_8.0.84-1_arm64.deb
$ sudo dpkg -i /home/nvidia/libcudnn6_6.0.21-1+cuda8.0_arm64.deb
$ sudo dpkg -i /home/nvidia/libcudnn6-dev_6.0.21-1+cuda8.0_arm64.deb
$ sudo dpkg -i /home/nvidia/libcudnn6-doc_6.0.21-1+cuda8.0_arm64.deb
$ sudo dpkg -i /home/nvidia/nv-gie-repo-ubuntu1604-ga-cuda8.0-trt2.1-20170614_1-1_arm64.deb
$ sudo apt update
$ sudo apt install cuda-toolkit-8.0
$ sudo apt install tensorrt-2.1.2
# add these commands at the end of `$HOME.profile`
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```




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
$ scp $HOME/nvidia/Linux_for_Tegra.tbz2 nvidia@192.168.42.71:/home/nvidia
```

