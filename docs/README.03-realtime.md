# Real-Time Linux for Jetson TX2
[Link to required NVIDIA packages](https://github.com/kozyilmaz/nvidia-jetson-rt#nvidia-jetson-tx2-configuration)

## Prepare build sources
```shell
$ mkdir -p $HOME/nvidia
$ cd $HOME/nvidia
$ tar -xvf gcc-4.8.5-aarch64.tgz
$ tar -xvf source_release.tbz2
$ tar -xvf sources/kernel_src-tx2.tbz2
```

## Create an `environment` file for envvars
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

## Export build variables and start compiling
Test kernel config can be found [here](https://github.com/kozyilmaz/nvidia-jetson-rt/raw/master/scripts/jetson-tx2-rt.config)!

```shell
$ cd $HOME/nvidia
$ source environment
$ mkdir -p $TEGRA_KERNEL_OUT
$ cd $HOME/nvidia/kernel/kernel-4.4
# list and apply real-time patches
$ for i in rt-patches/*.patch; do echo $i; done
$ for i in rt-patches/*.patch; do patch -p1 < $i; done
# create default config
$ make O=$TEGRA_KERNEL_OUT ARCH=$ARCH tegra18_defconfig
# change CONFIG_LOCALVERSION="-rt" and CONFIG_PREEMPT_RT_FULL=y for real-time scheduling
$ make O=$TEGRA_KERNEL_OUT ARCH=$ARCH menuconfig
# create compressed kernel image
$ make -j4 O=$TEGRA_KERNEL_OUT ARCH=$ARCH zImage
# compile device tree
$ make O=$TEGRA_KERNEL_OUT ARCH=$ARCH dtbs
# compile and install kernel modules
$ make -j4 O=$TEGRA_KERNEL_OUT ARCH=$ARCH modules
$ make O=$TEGRA_KERNEL_OUT ARCH=$ARCH modules_install INSTALL_MOD_PATH=$TEGRA_KERNEL_OUT/modules
```

## Copy binaries to `L4T` for deployment
```shell
$ mkdir -p $HOME/nvidia/L4T/kernel
$ cp $TEGRA_KERNEL_OUT/arch/arm64/boot/Image $HOME/nvidia/L4T/kernel
$ mkdir -p $HOME/nvidia/L4T/kernel/dtb
$ cp $TEGRA_KERNEL_OUT/arch/arm64/boot/dts/*.dtb $HOME/nvidia/L4T/kernel/dtb
$ cd $TEGRA_KERNEL_OUT/modules
$ tar --owner root --group root -cjf kernel_supplements.tbz2 *
$ cp $TEGRA_KERNEL_OUT/modules/kernel_supplements.tbz2 $HOME/nvidia/L4T/kernel/kernel_supplements.tbz2
$ cd $HOME/nvidia
$ tar -cjf $HOME/nvidia/L4T.tbz2 L4T
$ scp $HOME/nvidia/L4T.tbz2 nvidia@JETSON_IP_ADDRESS:/home/nvidia
```

## [TARGET] Update Kernel and Drivers on Jetson Board
```shell
$ cd /home/nvidia
$ tar -xjvf L4T.tbz2
$ sudo cp L4T/kernel/Image /boot/Image
$ sudo cp L4T/kernel/dtb/* /boot/dtb
$ sudo cp L4T/kernel/dtb/* /boot
$ sudo tar -xvf L4T/kernel/kernel_supplements.tbz2 -C /
$ sudo reboot
```
