#!/bin/bash

sudo ./RandomWalkExperiments/TX-max_perf.sh

taskset -c 3 chrt -f 1 ./RandomWalkExperiments/random_walk > random_walk_tm_fifo.txt &
#wait $!

taskset -c 3 chrt -f 1 ./RandomWalkExperiments/random_walk -zc > random_walk_zc_fifo.txt &
#wait $!

taskset -c 3 chrt -f 1 ./RandomWalkExperiments/random_walk -mm > random_walk_mm_fifo.txt &
#wait $!

taskset -c 3 chrt -f 1 ./InOrderWalkExperiments/inorder_walk > inorder_walk_tm_maphost.txt &
wait $!

taskset -c 3 chrt -f 1 ./InOrderWalkExperiments/inorder_walk -zc > inorder_walk_zc_maphost.txt &
wait $!

taskset -c 3 chrt -f 1 ./InOrderWalkExperiments/inorder_walk -mm > inorder_walk_mm_maphost.txt &
wait $!
