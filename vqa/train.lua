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

--------------
-- Training --
--------------
function runTrainVal()
    local step_trainval = false -- Train and Validaton
    local step_trainall = true -- Combine train2014 and val2014
    opt.save = paths.concat(opt.savepath,'model.t7')

    local stat = {}
    ------------------------------------
    -- Train and Validaton separately --
    ------------------------------------
    if step_trainval then
        local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014_train', 1)
        local state_val, _ = load_visualqadataset(opt, 'trainval2014_val', manager_vocab)
        local model, criterion = build_model(opt, manager_vocab)
        local paramx, paramdx = model:getParameters()
        local params_current, gparams_current = model:parameters()
        local config_layers, grad_last = config_layer_params(opt, params_current, gparams_current, 1)
        ------------------------
        -- Save local context --
        ------------------------
        local context = {
            model = model,
            criterion = criterion,
            paramx = paramx,
            paramdx = paramdx,
            params_current = params_current, 
            gparams_current = gparams_current,
            config_layers = config_layers,
            grad_last = grad_last
        }
        print('xxxxxxxxxxx Training started xxxxxxxxxxx ')
        for i = 1, opt.epochs do
            print('Epoch: '..i)
            train_epoch(opt, state_train, manager_vocab, context, 'train')
            -- per question type result stored
            _, _, perfs = train_epoch(opt, state_val, manager_vocab, context, 'val')
            -- Store statistics
            stat[i] = {acc, perfs.most_freq, perfs.openend_overall, perfs.multiple_overall}
            ------------------------------
            -- Reduce the learning rate --
            ------------------------------
            adjust_learning_rate(i, opt, config_layers)
        end
    end
    ----------------------------------
    -- Train and Validaton Combined --
    ----------------------------------
    if step_trainall then
        local nEpoch_best = 1
        local acc_openend_best = 0
        if step_trainval then
            -- Select the best train epoch number and combine train2014 and val2014
            for i = 1, #stat do
                if stat[i][3]> acc_openend_best then
                    nEpoch_best = i
                    acc_openend_best = stat[i][3]
                end
            end
          
            print('Best Epoch Number: ' .. nEpoch_best)
            print('Best Accuracy: ' .. acc_openend_best)
        else
            nEpoch_best = 1000
            -- max number of epochs to get the best epoch number from
            -- higher is better
            -- **make sure its not overtraining**
        end
        -- Combine train2014 and val2014
        local nEpoch_trainAll = nEpoch_best
        local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014', nil)
        -- recreate the model  
        local model, criterion = build_model(opt, manager_vocab)
        local paramx, paramdx = model:getParameters()
        local params_current, gparams_current = model:parameters()
        local config_layers, grad_last = config_layer_params(opt, params_current, gparams_current, 1)
        ------------------------
        -- Save local context --
        ------------------------
        local context = {
            model = model,
            criterion = criterion,
            paramx = paramx,
            paramdx = paramdx,
            params_current = params_current, 
            gparams_current = gparams_current,
            config_layers = config_layers,
            grad_last = grad_last
        }
        print('xxxxxxxxxxx Training started xxxxxxxxxxx ')
        stat = {}
        for i=1, nEpoch_trainAll do
            print('Epoch: '..i .. '/' .. nEpoch_trainAll)
            -- per question type result stored
            _, _, perfs = train_epoch(opt, state_train, manager_vocab, context, 'train')
            stat[i] = {acc, perfs.most_freq, perfs.openend_overall, perfs.multiple_overall}
            ------------------------------
            -- Reduce the learning rate --
            ------------------------------
            adjust_learning_rate(i, opt, config_layers)
            local modelname_curr = opt.save 
            save_model(opt, manager_vocab, context, modelname_curr)
        end
    end
end
runTrainVal()