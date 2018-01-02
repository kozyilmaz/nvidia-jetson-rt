#!/bin/bash
rm -f trace_gpu_*
nvprof --csv -u ms --print-gpu-trace --log-file trace_gpu_%p --profile-all-processes
