#!/usr/bin/env bash

# run 4 nodes
numNodes=3

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



th EASGD_server.lua --server --cuda --gpu 1 --numNodes $numNodes --numEpochs 50 --nodeIndex 0 --batchSize 128 --port $port --save testNet --host cnn-1404-titanx &
th EASGD_client.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 1 --batchSize 128 --port $port --host cnn-1404-titanx &
th EASGD_client.lua --cuda --gpu 2 --numNodes $numNodes --nodeIndex 2 --batchSize 128 --port $port --host cnn-1404-titanx &
# OMP_NUM_THREADS=28 th remote_temp.lua --numNodes $numNodes --nodeIndex 3 --batchSize 128 --port $port --host cnn-1404-titanx &

# run on a remote client
ssh -n -f lior@icri-lior "sh -c 'cd /home/lior/Playground/Torch/torch-distlearn/examples ; nohup /home/lior/torch/install/bin/th EASGD_client.lua --cuda --gpu 1 --numNodes $numNodes --nodeIndex 3 --batchSize 128 --port $port --host cnn-1404-titanx > /dev/null 2>&1 &'"

# run on a remote client-script example
# ssh -n -f lior@icri-lior "sh -c 'cd /home/lior/Playground/Torch/torch-distlearn/examples ; nohup ./remote_temp.sh $port > /dev/null 2>&1 &'"





# wait for them all
wait
