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
echo "$ sudo periodictaskreleaser/Benchmark/TX-max_perf.sh"
sudo periodictaskreleaser/Benchmark/TX-max_perf.sh

echo "\n=============================="
echo "Caffe Deep Learning Benchmarks"
echo "==============================\n"
echo "$ caffe/build/tools/caffe device_query -gpu 0"
caffe/build/tools/caffe device_query -gpu 0
echo "$ caffe/build/tools/caffe time --model=caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 200"
caffe/build/tools/caffe time --model=caffe/models/bvlc_alexnet/deploy.prototxt -gpu 0 -iterations 200
echo "$ caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 200"
caffe/build/tools/caffe time --model=caffe/models/bvlc_googlenet/deploy.prototxt -gpu 0 -iterations 200


echo "\n=================="
echo "Mixbench Benchmark"
echo "==================\n"
mkdir -p $BENCHMARK_FOLDER/mixbench
echo "$ mixbench/mixbench-cuda-ro > $BENCHMARK_FOLDER/mixbench/mixbench-cuda-ro.txt"
mixbench/mixbench-cuda-ro > $BENCHMARK_FOLDER/mixbench/mixbench-cuda-ro.txt
echo "$ mixbench/mixbench-cuda-alt > $BENCHMARK_FOLDER/mixbench/mixbench-cuda-alt.txt"
mixbench/mixbench-cuda-alt > $BENCHMARK_FOLDER/mixbench/mixbench-cuda-alt.txt


echo "\n================================"
echo "Periodic Task Releaser Benchmark"
echo "================================\n"
mkdir -p $BENCHMARK_FOLDER/periodictaskreleaser
echo "$ cd periodictaskreleaser/Benchmark"
cd periodictaskreleaser/Benchmark
echo "$ sudo ./TX-max_perf.sh"
sudo ./TX-max_perf.sh
echo "$ ruby run_benchmarks.rb"
ruby run_benchmarks.rb
echo "$ cp output.json ../../$BENCHMARK_FOLDER/periodictaskreleaser/"
cp output.json ../../$BENCHMARK_FOLDER/periodictaskreleaser/
echo "$ tar -czvf results.tar.gz results/"
tar -czvf results.tar.gz results/
echo "$ cp results.tar.gz ../../$BENCHMARK_FOLDER/periodictaskreleaser/"
cp results.tar.gz ../../$BENCHMARK_FOLDER/periodictaskreleaser/
echo "$ cp generate_plots.py ../../$BENCHMARK_FOLDER/periodictaskreleaser/"
cp generate_plots.py ../../$BENCHMARK_FOLDER/periodictaskreleaser/
echo "$ cd ../../"
cd ../../


echo "\n====================="
echo "CUDA Memory Benchmark"
echo "=====================\n"
mkdir -p $BENCHMARK_FOLDER/cudamemoryexperiments
echo "$ cd cudamemoryexperiments"
cd cudamemoryexperiments
echo "$ ./run_experiments.sh"
./run_experiments.sh
echo "$ cp *.txt  ../$BENCHMARK_FOLDER/cudamemoryexperiments/"
cp *.txt  ../$BENCHMARK_FOLDER/cudamemoryexperiments/
echo "$ cd ../"
cd ../


echo "$ tar -czvf $BENCHMARK_FOLDER.tar.gz $BENCHMARK_FOLDER"
tar -czvf $BENCHMARK_FOLDER.tar.gz $BENCHMARK_FOLDER
