#!/bin/sh

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

echo "\n=============================="
echo "Caffe Deep Learning Benchmarks"
echo "==============================\n"

echo "$ caffe/build/tools/caffe device_query -gpu 0"
caffe/build/tools/caffe device_query -gpu 0

echo "$ caffe/build/tools/caffe time --model=caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 100"
caffe/build/tools/caffe time --model=caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 100

echo "$ caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 100"
caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 100

