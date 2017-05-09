--this is a practice code similar to the mnsit stn code available
-- here https://github.com/qassemoquab/stnbhwd/tree/master/demo
-- I am using the distortion code as is and building the network

require 'nn'
require 'torch'
require 'distort_mnist'
require 'image'
require 'optim'
require 'xlua'
require 'cunn'
require 'stn'
-- getting distoreted dataset
print('creating distorted dataset')
trainDataset , testDataset = createDatasetsDistorted()

stn_true = true


function model()
	-- deciding parameters
	noutputs = 10
	height = 32
	width = 32
	depth = 1
	ninputs = height*width*depth
		--definition of convnet

	filterSize = 5
	stride = 1
	pad = 0
	poolSize = 2
	layerDepths = {64,64,128}

	--constructing the model

	model = nn.Sequential()

	if stn_true then
		dofile('spatial_transformer.lua')
		model:add(spanet)
	end
	model:add(nn.SpatialConvolution(depth,layerDepths[1],filterSize,filterSize,stride,stride))
	model:add(nn.ReLU())
	model:add(nn.SpatialConvolution(layerDepths[1],layerDepths[2],filterSize,filterSize,stride,stride))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(poolSize,poolSize,poolSize,poolSize))
	model:add(nn.SpatialConvolution(layerDepths[2],layerDepths[3],filterSize,filterSize,stride,stride))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.5))
	model:add(nn.View(layerDepths[3]*8*8))
	model:add(nn.Linear(layerDepths[3]*8*8,noutputs))
	model:add(nn.LogSoftMax())


	print('the model looks like =>')
	print(model)

	return model:cuda()

end
--function prepareDataset()
model = model()

function train(trainDataset)
	model:training()
	local time = sys.clock()

	criterion = nn.ClassNLLCriterion():cuda()
	--trainer = nn.StochasticGradient(model,criterion)
	function noOfEpochs() return 30 end

	for epoch = 1,noOfEpochs() do

		params,gradParams = model:getParameters()
		local optimState = {learningRate = 0.01,momentum = 0.9, weightDecay = 5e-4} 
		trainError = 0

		for batchIdx = 1,trainDataset:getNumBatches() do
			
			xlua.progress(batchIdx,trainDataset:getNumBatches())

			batchInputs,batchLabels = trainDataset:getBatch(batchIdx)
			batchInputs,batchLabels = batchInputs:cuda(),batchLabels:cuda()

			function feval(params)
				gradParams:zero()
				local outputs = model:forward(batchInputs)
				local loss = criterion:forward(outputs,batchLabels)
				local dloss_doutputs = criterion:backward(outputs,batchLabels)
				model:backward(batchInputs,dloss_doutputs)
				trainError = trainError + loss 
				return loss,gradParams
			end
			optim.sgd(feval,params,optimState)

					
		end
	print ('epoch : ',epoch,'trainError : ',trainError/trainDataset:getNumBatches())
	--torch.save('model')
	--return model
	end
end

print 'training'
train(trainDataset)
 
function test(testDataset)
	model:evaluate()
	criterion = nn.ClassNLLCriterion():cuda()
	--print 'check 0'
	local time = sys.clock()
	total = 0
	correct = 0
	
	for batchIdx = 1,testDataset:getNumBatches() do

		--xlua.progress(batchIdx,trainDataset:getNumBatches())
		inputs , labels = testDataset:getBatch(batchIdx)
		outputs = model:forward(inputs:cuda())
		--loss = criterion:forward(outputs:cuda(),labels())
		--testLoss = testLoss + loss
		_, output = outputs:max(2)
		output = output:cuda()
		correct = correct + output:eq(labels:cuda()):sum()
		total = total + output:size(1)	
	end

	print('test_accuracy : ',(correct/total)*100)

end

print 'testing'
test(testDataset)