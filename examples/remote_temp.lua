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
   --server                               Client/Server
   --verbose                               Print Communication details
   --communication     (default 10) How many batches between communications?
]]

-- Requires
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpu)
end

local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local Dataset = require 'dataset.Dataset'
local walkTable = require 'ipc.utils'.walkTable
local tau = opt.communication

require 'colorPrint' -- Print Server and Client in colors
if not opt.verbose then
  function printServer(string) end
  function printClient(string) end
end


-- Build the Network
local ipc = require 'libipc'
local Tree = require 'ipc.Tree'
local client, server
local serverBroadcast, clientBroadcast
if opt.server then
  print('client')
  serverBroadcast = ipc.server(opt.host, opt.port)
  serverBroadcast:clients(opt.numNodes, function (client) end)
  server = {}
  for i=1,opt.numNodes do
    client_port = opt.port + i
    print("Port #".. client_port .." for client #" .. i)
    server[i] = ipc.server(opt.host, client_port)
    server[i]:clients(1, function (client) end)
  end
else
  clientBroadcast = ipc.client(opt.host, opt.port)
  client = ipc.client(opt.host, opt.port + opt.nodeIndex)
end

local AsyncEA = require 'distlearn.AsyncEA'(server, serverBroadcast, client, clientBroadcast, opt.numNodes, opt.nodeIndex,tau, 0.2)

-- Print only in instance 0!
if not opt.verbose then
  if opt.nodeIndex > 0 then
     xlua.progress = function() end
     print = function() end
  end
end


-- Load the CIFAR-10 dataset
local lfs = require 'lfs'
dirPrefix = string.match(lfs.currentdir(),"/%a+/%a+/")
local trainingDataset = Dataset(dirPrefix .. 'Datasets/Cifar10/cifar10-train-twitter.t7', {
   -- Partition dataset so each node sees a subset:
   partition = opt.nodeIndex,
   partitions = opt.numNodes,
})

local testDataset = Dataset(dirPrefix .. 'Datasets/Cifar10/cifar10-test-twitter.t7', {
   -- Partition dataset so each node sees a subset:
   partition = opt.nodeIndex,
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


-- Train a neural network
AsyncEA.initClient(params)

for epoch = 1,100 do
  --  print('Training Epoch #'..epoch)
   for i = 1,numTrainingBatches() do
      -- Next sample:
      local batch = getTrainingBatch()
      local x = batch.input
      local y = batch.target

      -- Grads:
      -- print(opt.nodeIndex .. " Computing")


      local grads, loss, prediction = df(params,x,y)

      -- sync with Master
      AsyncEA.syncClient(params) -- Syncs client with server if needed


      -- Update weights and biases
      for layer in pairs(params) do
         for i in pairs(params[layer]) do
            params[layer][i]:add(-opt.learningRate, grads[layer][i])
         end
      end

   end

end
