#!/usr/bin/env bash

# run 4 nodes
fuser -k 8080/tcp
OMP_NUM_THREADS=4 th client_remote.lua --numNodes 2 --nodeIndex 1 --batchSize 256 &
th client_remote.lua --numNodes 2 --nodeIndex 2 --batchSize 256 --cuda --gpu 1 &


# th remote.lua --numNodes 3 --nodeIndex 2 --batchSize 256 --cuda --gpu 2 &
# th client_remote.lua --cuda --gpu 1 --numNodes 3 --base 3 --nodeIndex 3 --batchSize 384 --host 10.185.36.35 --port 8040 &
# th remote.lua --numNodes 4 --nodeIndex 3 --batchSize 512 --base 4 --host 10.185.36.45 port 15200 &
# th remote.lua --numNodes 4 --nodeIndex 4 --batchSize 512 --base 4 --host 10.185.36.45 port 15200 &

# wait for them all
wait
