
#### NVIDIA Jetson TX2 CPU Capabilities
```shell
$ cat /proc/cpuinfo 
processor	: 0
model name	: ARMv8 Processor rev 3 (v8l)
BogoMIPS	: 62.50
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0xd07
CPU revision	: 3

processor	: 3
model name	: ARMv8 Processor rev 3 (v8l)
BogoMIPS	: 62.50
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0xd07
CPU revision	: 3

processor	: 4
model name	: ARMv8 Processor rev 3 (v8l)
BogoMIPS	: 62.50
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0xd07
CPU revision	: 3

processor	: 5
model name	: ARMv8 Processor rev 3 (v8l)
BogoMIPS	: 62.50
Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0xd07
CPU revision	: 3
```

#### NVIDIA Jetson TX2 GPU Capabilities
```shell
$ ./deviceQuery 
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA Tegra X2"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    6.2
  Total amount of global memory:                 7842 MBytes (8223334400 bytes)
  ( 2) Multiprocessors, (128) CUDA Cores/MP:     256 CUDA Cores
  GPU Max Clock rate:                            1301 MHz (1.30 GHz)
  Memory Clock rate:                             1600 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 524288 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 32768
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            Yes
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = NVIDIA Tegra X2
Result = PASS
```

#### Caffe Device Query
```shell
$ caffe/build/tools/caffe device_query -gpu 0
I0102 16:37:09.743705 19642 caffe.cpp:138] Querying GPUs 0
I0102 16:37:09.750903 19642 common.cpp:178] Device id:                     0
I0102 16:37:09.750962 19642 common.cpp:179] Major revision number:         6
I0102 16:37:09.750980 19642 common.cpp:180] Minor revision number:         2
I0102 16:37:09.750993 19642 common.cpp:181] Name:                          NVIDIA Tegra X2
I0102 16:37:09.751010 19642 common.cpp:182] Total global memory:           8232349696
I0102 16:37:09.751026 19642 common.cpp:183] Total shared memory per block: 49152
I0102 16:37:09.751040 19642 common.cpp:184] Total registers per block:     32768
I0102 16:37:09.751054 19642 common.cpp:185] Warp size:                     32
I0102 16:37:09.751063 19642 common.cpp:186] Maximum memory pitch:          2147483647
I0102 16:37:09.751075 19642 common.cpp:187] Maximum threads per block:     1024
I0102 16:37:09.751086 19642 common.cpp:188] Maximum dimension of block:    1024, 1024, 64
I0102 16:37:09.751098 19642 common.cpp:191] Maximum dimension of grid:     2147483647, 65535, 65535
I0102 16:37:09.751108 19642 common.cpp:194] Clock rate:                    1300500
I0102 16:37:09.751121 19642 common.cpp:195] Total constant memory:         65536
I0102 16:37:09.751132 19642 common.cpp:196] Texture alignment:             512
I0102 16:37:09.751143 19642 common.cpp:197] Concurrent copy and execution: Yes
I0102 16:37:09.751157 19642 common.cpp:199] Number of multiprocessors:     2
I0102 16:37:09.751165 19642 common.cpp:200] Kernel execution timeout:      No
```

#### Throttle Script
````shell
$ cat TX-max_perf.sh 
#!/bin/sh
echo "WARNING - Must Be Run Sudo"

echo "WARNING - Use Only on TX2"

echo "Turn on fan for safety"
echo 255 > /sys/kernel/debug/tegra_fan/target_pwm
echo "Fan setting"
cat /sys/kernel/debug/tegra_fan/target_pwm

echo "Cores active"
cat /sys/devices/system/cpu/online

echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor

echo "Scaling governors (0, 3, 4, 5)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor

echo "CPU available frequencies"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu5/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq

echo "CPU minimum cycle frequencies (0, 3, 4, 5)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu5/cpufreq/scaling_min_freq

echo "Max Performance Settings Done"
```
