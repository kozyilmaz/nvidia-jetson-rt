CUDA Microbenchmarks
====================

About
-----

This repository contains a small collection of microbenchmarks for profiling
CUDA programs. Most focus on memory accesses.

Usage
-----

Building this requires CUDA. Also, the code is intended to be used on the TX1
board, but it can probably run on any system which supports CUDA.

To run benchmarks in this directory:
```bash
# Increase clock rates and disable frequency scaling (TX1 only!)
sudo ./TX-max-perf.sh

# Compile and run the benchmark
make
./random_walk
```
