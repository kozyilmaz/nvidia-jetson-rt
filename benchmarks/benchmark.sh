#!/bin/sh

echo "\n============================"
echo "NVIDIA Jetson TX2 Benchmarks"
echo "============================\n"
BENCHMARK_FOLDER=benchmark-`uname -r`
mkdir -p $BENCHMARK_FOLDER

echo "\n====================================="
echo "NVIDIA Jetson TX2 Kernel Capabilities"
echo "=====================================\n"
echo "$ uname -a"
uname -a
echo "$ zcat /proc/config.gz |grep CONFIG_HZ"
zcat /proc/config.gz |grep CONFIG_HZ
echo "$ zcat /proc/config.gz |grep CONFIG_PREEMPT"
zcat /proc/config.gz |grep CONFIG_PREEMPT

echo "\n=============================="
echo "NVIDIA Jetson TX2 CPU & Memory"
echo "==============================\n"
echo "$ cat /proc/cpuinfo"
cat /proc/cpuinfo
echo "$ cat /proc/meminfo"
cat /proc/meminfo

echo "\n=========================="
echo "NVIDIA Jetson TX2 Throttle"
echo "==========================\n"

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


echo "\n=============================="
echo "Caffe Deep Learning Benchmarks"
echo "==============================\n"
echo "$ caffe/build/tools/caffe device_query -gpu 0"
caffe/build/tools/caffe device_query -gpu 0
echo "$ caffe/build/tools/caffe time --model=caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 200"
#caffe/build/tools/caffe time --model=caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 200
echo "$ caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 200"
#caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 200


echo "\n================================"
echo "Periodic Task Releaser Benchmark"
echo "================================\n"
mkdir -p $BENCHMARK_FOLDER/periodictaskreleaser
echo "$ cd periodictaskreleaser/Benchmark"
cd periodictaskreleaser/Benchmark
echo "$ ruby run_benchmarks.rb"
ruby run_benchmarks.rb
echo "$ cp output.json ../../$BENCHMARK_FOLDER/periodictaskreleaser/"
cp output.json ../../$BENCHMARK_FOLDER/periodictaskreleaser/
echo "$ cp results.tar.gz ../../$BENCHMARK_FOLDER/periodictaskreleaser/"
cp results.tar.gz ../../$BENCHMARK_FOLDER/periodictaskreleaser/
echo "$ cp generate_plots.py ../../$BENCHMARK_FOLDER/periodictaskreleaser/"
cp generate_plots.py ../../$BENCHMARK_FOLDER/periodictaskreleaser/
echo "$ cd ../../"
cd ../../


