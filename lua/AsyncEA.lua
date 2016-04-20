
local ipc = require 'libipc'
local walkTable = require 'ipc.utils'.walkTable


local function AsyncEA(server, serverBroadcast, client, clientBroadcast,numNodes, node, tau, alpha)

   -- Keep track of how many steps each node does per epoch
   local step = 0

   -- Keep track of the center point (also need space for the delta)
   local center,delta,flatParam

   local currentClient
   local busyFlag = 0

   -- Clone the parameters to use a center point
   local function oneTimeInit(params)
      if not center then
         center = { }
         delta = { }
         flatParam = { }
         walkTable(params, function(param)
            table.insert(center, param:clone())
            table.insert(delta, param:clone())
            table.insert(flatParam, param)
         end)
      end
   end

   local function initServer(params)

     oneTimeInit(params)

     serverBroadcast:clients(function(client)
       walkTable(center, function(valuei)
         client:send(valuei)
       end)
     end)
   end

   local function initClient(params)

     oneTimeInit(params)

     walkTable(center, function(valuei)
       return clientBroadcast:recv(valuei)
     end)
     local i = 1
     walkTable(params, function(param)
        param:copy(center[i])
        i = i + 1
     end)
   end


   -- Average the parameters according to http://arxiv.org/abs/1412.6651
   local function averageParameters(params)
      -- First time we need to initialize the center point and delta
      oneTimeInit(params)
      -- This node contributed to this step
      step = step + 1
      -- If its time to run an average
      if step % tau == 0 then
         -- Compute our elastic difference (delta)
         -- and move this node towards the center point
         local i = 1
         walkTable(params, function(param)
            delta[i]:add(param, -1, center[i]):mul(alpha)
            param:add(-1, delta[i])
            i = i + 1
         end)
         -- AllReduce the elastic differences
        --  tree.allReduce(delta, function(a, b) return a:add(b) end)
         -- Move the center point towards the nodes
         for i = 1,#center do
            center[i]:add(delta[i])
         end
      end
   end

   -- Do some fanciness to get all the nodes to the same point
  --  local function handleUnevenSteps(params)
  --     -- Only need to synchronize nodes if we have done at least one step
  --     if step > 0 then
  --        -- Wow, this is expressed terribly
  --        -- Do one final all reduce to get all the nodes in sync
  --        for i = 1,#delta do
  --           delta[i]:fill(0)
  --        end
  --        tree.allReduce(delta,
  --           function(a, b) return a:add(b) end,
  --           function(_, i)
  --              -- Move the center point towards the nodes
  --              center[i]:add(delta[i])
  --              -- Compute our elastic difference (delta)
  --              -- and move this node towards the center point
  --              delta[i]:add(flatParam[i], -1, center[i]):mul(alpha)
  --              flatParam[i]:add(-1, delta[i])
  --              return delta[i]
  --           end)
  --        -- Reset step counter
  --        step = 0
  --     end
  --  end
   --
  --  -- Ensure the same exact center point is on every node
  --  -- Call at the end of epoch (or at any point you desire)
  --  -- Over time the center points will drift a bit due to floating point error accumulation
  --  local function synchronizeCenter(params)
  --     -- First time we need to initialize the center point and delta
  --     oneTimeInit(params)
  --     -- Handle uneven # of steps per node
  --     handleUnevenSteps(params)
  --     -- Scatter the center point
  --     tree.scatter(center)
  --  end

  --  -- At any point in time you can force the same parameters on all nodes
  --  local function synchronizeParameters(params)
  --     -- First time we need to initialize the center point and delta
  --     oneTimeInit(params)
  --     -- Handle uneven # of steps per node
  --     handleUnevenSteps(params)
  --     -- Scatter the parameters
  --     tree.scatter(params)
  --     -- Reset the center to the parameters
  --     local i = 1
  --     tree.walkTable(params, function(param)
  --        center[i]:copy(param)
  --        i = i + 1
  --     end)
  --  end

   -- Client Sends

   local tempValue
   local function getTempValue(value)
     if torch.isTensor(value) then
        if tempValue then
           tempValue = tempValue:typeAs(value):resizeAs(value)
        else
           tempValue = value:clone()
        end
        return tempValue
     end
   end




  local function isSyncNeeded()

    step = step + 1

    if step % 10 == 0 then
      return true
    end

    return false

  end


  local function clientEnterSync()

    printClient(node,"Waiting to sync")
    clientBroadcast:send({ q = "Enter?",
      clientID = node})
    assert(client:recv() == "Enter")
    printClient(node,"Entered Sync")


   end

   local function serverEnterSync()

     printServer("Server waiting to sync")

     msg = serverBroadcast:recvAny()
     assert(msg.q == "Enter?")
     currentClient = msg.clientID
     printServer("Current client is #" .. currentClient)
     server[currentClient]:clients(1, function(client)
       client:send("Enter")
     end)


    end


  local function clientGetCenter(params)

    client:send("Center?")

    walkTable(center, function(valuei)
     return client:recv(valuei)
    end)

    printClient(node,"Received center")

    end

    local function serverSendCenter(params)

      local function serverHandler(client)

        local msg = client:recv()
        assert(msg == "Center?")
        printServer("Client #" .. currentClient)
        walkTable(center, function(valuei)
          client:send(valuei)
          end)
        printServer("Server Sent Center to Client #" .. currentClient)
      end

      server[currentClient]:clients(1, serverHandler)
  end

   local function calculateUpdateDiff(params)
     local i = 1
     walkTable(params, function(param)
        delta[i]:add(param, -1, center[i]):mul(alpha)
        param:add(-1, delta[i])
        i = i + 1
     end)
   end

   local function clientSendDiff(params)

     client:send("delta?")
     assert(client:recv() == "delta")
     printClient(node,"Received ack for sending delta")
     walkTable(delta, function(valuei)
     client:send(valuei)
     end)
    end

  local function serverGetUpdateDiff(params)

    local function GetUpdateDiffHandler(client)

      assert(client:recv() == "delta?")
      client:send("delta")

      walkTable(delta, function(valuei)
        return client:recv(valuei)
      end)          -- update server-master node

      printServer("Received delta from client #" .. currentClient)

      local i = 1
      walkTable(center, function(param)
         param:add(delta[i])
         i = i + 1
      end)

    end

    server[currentClient]:clients(1, GetUpdateDiffHandler)

    local i = 1
    walkTable(params, function(param)
      param:copy(center[i])
      i = i + 1
    end)

  end


  local function syncClient(params)

    if isSyncNeeded() then
      clientEnterSync() -- Start communication if needed with server, stand in line
      clientGetCenter(params) -- Receive required parameters from the server
      calculateUpdateDiff(params) -- Do calculations locally
      clientSendDiff(params) -- Send updated values to the server
      return true -- if synced
    end

    return false

  end

  local function syncServer(params)

    serverEnterSync() -- Enter critical section. Allow only one client connection
    serverSendCenter(params) -- Send required parameters to client
    serverGetUpdateDiff(params) -- Get parameters from client and update server

  end

 return {
      initServer = initServer,
      initClient = initClient,
      syncClient = syncClient,
      syncServer = syncServer
   }
end

return AsyncEA
