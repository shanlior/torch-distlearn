#!/usr/bin/env bash

# run 4 nodes

# kill gpu lua
# kill GPU PID $(nvidia-smi -g 0 | awk '$2=="Processes:" {p=1} p && $3 > 0 && $5~/luajit/ {print $3}')


if [ -z "$1" ]
  then
  port=`echo 8080`
else
  port=$1
fi

numNodes=2

currPort=$port
for i in `seq 0 $numNodes`;
do
  fuser -k $currPort/tcp
  # echo Kill port $currPort
  currPort=$(($currPort + 1))
done

# OMP_NUM_THREADS=4 th remote_temp_server.lua --server 1 --numNodes 1 --nodeIndex 0 --batchSize 128 --port $port --verbose 0 &
# th remote_temp.lua --numNodes 1 --nodeIndex 1 --batchSize 128 --cuda --gpu 1 --port $port --verbose 0 &

# th remote_temp_server.lua  --cuda --gpu 1 --server 1 --numNodes 2 --nodeIndex 0 --batchSize 128 --port $port --save testNet &
# th remote_temp.lua --cuda --gpu 1 --numNodes 2 --nodeIndex 1 --batchSize 128 --port $port &
th remote_temp.lua --cuda --gpu 1 --numNodes 4 --nodeIndex 4 --batchSize 128 --port $port --host 10.185.36.35 &



# th remote.lua --numNodes 3 --nodeIndex 2 --batchSize 256 --cuda --gpu 2 &
# th client_remote.lua --cuda --gpu 1 --numNodes 3 --base 3 --nodeIndex 3 --batchSize 384 --host 10.185.36.35 --port 8040 &
# th remote.lua --numNodes 4 --nodeIndex 3 --batchSize 512 --base 4 --host 10.185.36.45 port 15200 &
# th remote.lua --numNodes 4 --nodeIndex 4 --batchSize 512 --base 4 --host 10.185.36.45 port 15200 &

# wait for them all
wait
