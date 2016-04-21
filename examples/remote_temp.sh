#!/usr/bin/env bash

# run 4 nodes
numNodes=4

#################################################
# Try to close the ports if they are already used

if [ -z "$1" ]
  then
  port=`echo 8080`
else
  port=$1
fi


currPort=$port
for i in `seq 0 $numNodes`;
do
  fuser -k $currPort/tcp
  # echo Kill port $currPort
  currPort=$(($currPort + 1))
done

# OPTIONAL: Uncomment if you want to close all luajit Processes running on GPU's
# kill $(nvidia-smi -g 0 | awk '$2=="Processes:" {p=1} p && $3 > 0 && $5~/luajit/ {print $3}')


#################################################

th remote_temp.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 4 --batchSize 128 --port $port --host 10.185.36.35 &


# wait for them all
wait
