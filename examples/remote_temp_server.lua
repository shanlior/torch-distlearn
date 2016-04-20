local opt = lapp [[
Train a CNN classifier on CIFAR-10 using AllReduceSGD.

   --nodeIndex         (default 1)         node index
   --numNodes          (default 1)         num nodes spawned in parallel
   --batchSize         (default 32)        batch size, per node
   --learningRate      (default .01)        learning rate
   --cuda                                  use cuda
   --gpu               (default 1)         which gpu to use (only when using cuda)
   --host              (default '127.0.0.1') host name of the server
   --port              (default 8080)      port number of the server
   --base              (default 2)         power of 2 base of the tree of nodes
   --clientIP          (default '127.0.0.1') host name of the client
   --server                                 Client/Server
   --verbose                                Print Communication details
   --communication     (default 10) How many batches between communications?
   --save              (default 'log') Save location
   --visualise         (default 1)

]]

-- Requires
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpu)
end
-- luarocks install autograd
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local Dataset = require 'dataset.Dataset'

if opt.save == 'log' then
  opt.save = os.date():gsub(' ','')
end

opt.save = paths.concat('./Results', opt.save)
os.execute('mkdir -p ' .. opt.save)
local cmd = torch.CmdLine()
cmd:log(opt.save .. '/Log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)


require 'colorPrint' -- Print Server and Client in colors
if not opt.verbose then
  function printServer(string) end
  function printClient(string) end
end
-- Build the AllReduce tree


local ipc = require 'libipc'
local Tree = require 'ipc.Tree'
local client, server
local serverBroadcast, clientBroadcast
if opt.server then
  serverBroadcast = ipc.server(opt.host, opt.port)
  serverBroadcast:clients(opt.numNodes, function (client) end)
  server = {}
  for i=1,opt.numNodes do
    client_port = opt.port + i
    printServer("Port #".. client_port .." for client #" .. i)
    server[i] = ipc.server(opt.host, client_port)
    server[i]:clients(1, function (client) end)
  end
else
  clientBroadcast = ipc.client(opt.host, opt.port)
  client = ipc.client(opt.host, opt.port + opt.nodeIndex)
end
-- local tree =  Tree(opt.nodeIndex, opt.numNodes, opt.base, server, client, opt.clientIP, opt.port + opt.nodeIndex)

local AsyncEA = require 'distlearn.AsyncEA'(server, serverBroadcast, client, clientBroadcast,opt.numNodes, 1,10, 0.2)

-- Print only in instance 0!
if not opt.server then
   xlua.progress = function() end
   print = function() end
end

-- Adapt batch size, per node:
-- if not opt.cuda then
--   print('CPU mode')
--   opt.batchSize = math.ceil(opt.batchSize / 16)
-- end
-- opt.batchSize = math.ceil(opt.batchSize / opt.numNodes)
printServer('Batch size: per node = ' .. opt.batchSize .. ', total = ' .. (opt.batchSize*opt.numNodes))




-- Load the CIFAR-10 dataset
-- trainData = torch.load('/home/ehoffer/Datasets/Cifar10/cifar10-train.t7')
local trainingDataset = Dataset('/home/lior/Datasets/Cifar10/cifar10-train-twitter.t7', {
   -- Partition dataset so each node sees a subset:
   partition = 1,
   partitions = opt.numNodes,
})

local testDataset = Dataset('/home/lior/Datasets/Cifar10/cifar10-test-twitter.t7', {
   -- Partition dataset so each node sees a subset:
   partition = 1,
   partitions = opt.numNodes,
})

local getTrainingBatch, numTrainingBatches = trainingDataset.sampledBatcher({
   samplerKind = 'label-uniform',
   batchSize = opt.batchSize,
   inputDims = { 3, 32, 32 },
   verbose = true,
   cuda = opt.cuda,
   processor = function(res, processorOpt, input)
      -- This function is not a closure, it is run in a clean Lua environment
      local image = require 'image'
      -- Turn the res string into a ByteTensor (containing the PNG file's contents)
      -- local bytes = torch.ByteTensor(#res)
      -- bytes:storage():string(res)
      -- Decompress the PNG bytes into a Tensor
      -- local pixels = image.decompressPNG(bytes)
      -- Copy the pixels tensor into the mini-batch
      input:copy(res)
      return true
   end,
})


local getTestBatch, numTestBatches = testDataset.sampledBatcher({
   samplerKind = 'label-uniform',
   batchSize = opt.batchSize,
   inputDims = { 3, 32, 32 },
   verbose = true,
   cuda = opt.cuda,
   processor = function(res, processorOpt, input)
      -- This function is not a closure, it is run in a clean Lua environment
      local image = require 'image'
      -- Turn the res string into a ByteTensor (containing the PNG file's contents)
      -- local bytes = torch.ByteTensor(#res)
      -- bytes:storage():string(res)
      -- Decompress the PNG bytes into a Tensor
      -- local pixels = image.decompressPNG(bytes)
      -- Copy the pixels tensor into the mini-batch
      input:copy(res)
      return true
   end,
})
-- Load in MNIST
local classes = {
   'airplane', 'automobile', 'bird', 'cat',
   'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
}
local confusionMatrix = optim.ConfusionMatrix(classes)

-- for CNNs, we rely on efficient nn-provided primitives:
local conv,params,bn,acts,pool = {},{},{},{},{}
local flatten,linear

-- Ensure same init on all nodes:
torch.manualSeed(0)

-- layer 1:
conv[1], params[1] = grad.nn.SpatialConvolutionMM(3, 64, 5,5, 1,1, 2,2)
bn[1], params[2] = grad.nn.SpatialBatchNormalization(64, 1e-3)
acts[1] = grad.nn.ReLU()
pool[1] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 2:
conv[2], params[3] = grad.nn.SpatialConvolutionMM(64, 128, 5,5, 1,1, 2,2)
bn[2], params[4] = grad.nn.SpatialBatchNormalization(128, 1e-3)
acts[2] = grad.nn.ReLU()
pool[2] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 3:
conv[3], params[5] = grad.nn.SpatialConvolutionMM(128, 256, 5,5, 1,1, 2,2)
bn[3], params[6] = grad.nn.SpatialBatchNormalization(256, 1e-3)
acts[3] = grad.nn.ReLU()
pool[3] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 4:
conv[4], params[7] = grad.nn.SpatialConvolutionMM(256, 512, 5,5, 1,1, 2,2)
bn[4], params[8] = grad.nn.SpatialBatchNormalization(512, 1e-3)
acts[4] = grad.nn.ReLU()
pool[4] = grad.nn.SpatialMaxPooling(2,2, 2,2)

-- layer 5:
flatten = grad.nn.Reshape(512*2*2)
linear,params[9] = grad.nn.Linear(512*2*2, 10)

-- Cast the parameters
params = grad.util.cast(params, opt.cuda and 'cuda' or 'float')

-- Make sure all the nodes have the same parameter values
-- AsyncEA.synchronizeParameters(params)

-- Loss:
local logSoftMax = grad.nn.LogSoftMax()
local crossEntropy = grad.nn.ClassNLLCriterion()

-- Define our network
local function predict(params, input, target)
   local h = input
   local np = 1
   for i in ipairs(conv) do
      h = pool[i](acts[i](bn[i](params[np+1], conv[i](params[np], h))))
      np = np + 2
   end
   local hl = linear(params[np], flatten(h), 0.5)
   local out = logSoftMax(hl)
   return out
end

-- Define our loss function
local function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = crossEntropy(prediction, target)
   return loss, prediction
end

-- Get the gradients closure magically:
local df = grad(f, {
   optimize = true,              -- Generate fast code
   stableGradients = true,       -- Keep the gradient tensors stable so we can use CUDA IPC
})


AsyncEA.initServer(params)

local epoch = 0
-- Train a neural network
for syncID = 1,10000 do

  AsyncEA.syncServer(params)

  print('Sync #' .. syncID)
  xlua.progress(syncID % 100, 100)

  if syncID % 100 == 0 then -- every 100 syncs test the net

    epoch = epoch + 1

    -- Check Training Error
    print('Training Error Trial #'..epoch)

    for i = 1,numTrainingBatches() do
       -- Next sample:
       local batch = getTrainingBatch()
       local x = batch.input
       local y = batch.target

       -- Prediction:
       local loss, prediction = f(params,x,y)

       -- Log performance:
       for b = 1,batch.batchSize do
          confusionMatrix:add(prediction[b], y[b])
       end

       -- Display progress:
       xlua.progress(i, numTestBatches())
    end

    print(confusionMatrix)
    local ErrTrain = (1-confusionMatrix.totalValid)
    print('Training Error = ' .. ErrTrain)
    confusionMatrix:zero()

    -- Check Test Error
    print('Testing Error Trial #' ..epoch)


    for i = 1,numTestBatches() do
      -- Next sample:
      local batch = getTestBatch()
      local x = batch.input
      local y = batch.target

      -- Prediction:
      local loss, prediction = f(params,x,y)
      -- -- Gather the grads from all nodes
      -- allReduceSGD.sumAndNormalizeGradients(grads)

      -- Log performance:
      for b = 1,batch.batchSize do
         confusionMatrix:add(prediction[b], y[b])
      end

      -- Display progress:
      xlua.progress(i, numTestBatches())
    end

    print(confusionMatrix)
    local ErrTest = (1-confusionMatrix.totalValid)
    print('Test Error = ' .. ErrTest)
    confusionMatrix:zero()

    Log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
    if opt.visualize == 1 then
        Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
        Log:plot()
    end

  end
end
