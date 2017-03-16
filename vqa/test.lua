require 'paths'
require 'cunn'
require 'nn'

local stringx = require 'pl.stringx'
local file = require 'pl.file'
local debugger = require 'fb.debugger'
local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end

paths.dofile('utils.lua')

-------------
-- Testing --
-------------
function runTest()
    print('Testing')
    local model_path = 'models/iBOWIMG-2x.t7'
    local testSet = 'test2015' -- 'test-dev2015'
    local f_model = torch.load(model_path)
    local manager_vocab = f_model.manager_vocab
    local model, criterion = build_model(opt, manager_vocab)
    local paramx, paramdx = model:getParameters()
    paramx:copy(f_model.paramx)
    local state_test, _ = load_visualqadataset(opt, testSet, manager_vocab)
    local context = {
        model = model,
        criterion = criterion,
    }

    print('xxxxxxxxx PREDICTIONS xxxxxxxxx')
    local pred, prob, perfs = train_epoch(opt, state_test, manager_vocab, context, 'test')
    local file_json_multiple = 'MultipleChoice_' .. testSet .. '_'.. 'iBOWIMG-2x' .. '_results.json'
    print('Wrote the MultipleChoice prediction to JSON file...'..file_json_multiple) 
    local choice = 1
    outputJSONanswer(state_test, manager_vocab, prob, file_json_multiple, choice)
    collectgarbage()

end


runTest()