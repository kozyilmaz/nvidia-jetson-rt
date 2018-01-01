# Building a Jetson TX2 image and flashing it to eMMC

## Creating NVIDIA Jetson TX2 image
```shell
$ mkdir -p $HOME/nvidia/docs
$ tar -xvf NVIDIA_Tegra_Linux_Driver_Package.tar -C $HOME/nvidia/docs
$ sudo tar -xvf Tegra186_Linux_R28.1.0_aarch64.tbz2
$ sudo tar -xvf Tegra_Linux_Sample-Root-Filesystem_R28.1.0_aarch64.tbz2 -C $HOME/nvidia/Linux_for_Tegra/rootfs
$ cd $HOME/nvidia/Linux_for_Tegra
$ sudo ./apply_binaries.sh
```

## Flashing NVIDIA Jetson TX2 internal eMMC
![Screenshot](https://github.com/kozyilmaz/nvidia-jetson-rt/raw/master/docs/console.jpg "NVIDIA Jetson TX2 Console Connection")
```shell
# put board in recovery mode
Connect Power Unit
Press POWER
Release POWER
Push RECOVERY_FORCE
Push RESET (after one second)
Release RESET
Release RECOVERY_FORCE (after two second)

# run flash command
$ sudo ./flash.sh -t jetson-tx2 mmcblk0p1
```

