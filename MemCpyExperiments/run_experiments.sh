#!/bin/bash

# Argument 1: program name
# Argument 2: output file name
# Argument 3: zero copy

output=$1

for i in `seq 10 30`;
do
  num=$((2**$i))
  nvprof --profile-api-trace none --log-file tmp.txt ./memcpy ${num} &
  pid=$!
  wait ${pid}
  #cat tmp.txt | grep "DoRandomWalk" >> ${output}
  cat tmp.txt | grep "HtoD" >> ${output}
done


