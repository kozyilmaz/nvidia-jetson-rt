#!/bin/bash

# Log CPU and memory usage of a process.
# Parameter 1: pid.
# Paramter 2: output file.
# Parameter 3: Program name (for logging information only).

pid=$1
outfile=$2
pname=$3
echo "Timestamp, program, PID, %cpu, rss" > $outfile # erase the output file
while kill -0 $pid 2>/dev/null;
do
  echo -n "$(date +%s.%N), ${pname}, ${pid}, " >> $outfile
  stdbuf -oL ps --no-headers -p $pid -o %cpu,rss | sed 's/^ *//g' | tr -s " " | cut --delimiter=" " --output-delimiter="," --fields=1,2 >> $outfile
  sleep 2.0
done

