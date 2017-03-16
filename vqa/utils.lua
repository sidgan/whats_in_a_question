require 'nn'
require 'cunn'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end

paths.dofile('LinearNB.lua')
local stringx = require 'pl.stringx'
local file = require 'pl.file'
local debugger = require 'fb.debugger'

function getFreeGPU()
    -- select the most available GPU to train
    local nDevice = cutorch.getDeviceCount()
    local memSet = torch.Tensor(nDevice)
    for i=1, nDevice do
        local tmp, _ = cutorch.getMemoryUsage(i)
        memSet[i] = tmp
    end
    local _, curDeviceID = torch.max(memSet,1)
    return curDeviceID[1]
end

---------------------------
-- Make iBOWIMG-2x model --
---------------------------
function build_model(opt, manager_vocab)
    model = nn.Sequential()
    local module_tdata = nn.Sequential():add(nn.SelectTable(1)):add(nn.LinearNB(manager_vocab.nvocab_question, opt.embed_word))
    local module_tdata_other = nn.Sequential():add(nn.SelectTable(1)):add(nn.LinearNB(manager_vocab.nvocab_question, opt.embed_word))
    --[[ For empty tensor
    local module_tdata_other = = torch.Tensor(opt.embed_word)
    ]]
    local module_vdata = nn.Sequential():add(nn.SelectTable(2))
    local cat = nn.ConcatTable():add(module_tdata_other):add(module_tdata):add(module_vdata)
    model:add(cat):add(nn.JoinTable(2))
    model:add(nn.LinearNB(opt.embed_word  + opt.embed_word + opt.vdim, manager_vocab.nvocab_answer))
    model:add(nn.Normalize(1))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
    model:cuda()
    criterion:cuda()
    return model, criterion
end

------------------------------
-- Reduce the learning rate --
------------------------------
function adjust_learning_rate(epoch_num, opt, config_layers)
    if epoch_num % opt.nepoch_lr == 0 then
        for j = 1, #config_layers.lr_rates do
            config_layers.lr_rates[j] = config_layers.lr_rates[j] / opt.decay
        end
    end
end

------------------------------
--- Load pre-trained model ---
---- Evaluate on test set ---- 
----- Generate csv file ------
------------------------------
function loadPretrained(opt)
    local model_path = 'model/iBOWIMG-2x.t7'
    local f_model = torch.load(model_path)
    local manager_vocab = f_model.manager_vocab 
    -- apply fix
    if manager_vocab.vocab_map_question['END'] == nil then 
        manager_vocab.vocab_map_question['END'] = -1 
        manager_vocab.ivocab_map_question[-1] = 'END'
    end
    local model, criterion = build_model(opt, manager_vocab)
    local paramx, paramdx = model:getParameters()
    paramx:copy(f_model.paramx)
    return {
        model = model,
        criterion = criterion,
        manager_vocab = manager_vocab
    }
end

------------------------------
-- LR and clipping gradient --
------------------------------
function config_layer_params(opt, params_current, gparams_current, IDX_wordembed)
    local lr_wordembed = opt.lr_wordembed
    local lr_other = opt.lr_other
    local weightClip_wordembed = opt.weightClip_wordembed
    local weightClip_other = opt.weightClip_other

    print("Learning Rate for Word Embeddings = " .. lr_wordembed)
    print("Learning Rate for Training = " .. lr_other)
    print("Weight Clip for Word Embeddings = " .. weightClip_wordembed)
    print("Weight Clip for Training = " .. weightClip_other)

    local gradientClip_dummy = 0.1
    local weightRegConsts_dummy = 0.000005
    local initialRange_dummy = 0.1
    local moments_dummy = 0.9

    -- Initialize specification of layers
    local config_layers = {
        lr_rates = {},
        gradientClips = {},
        weightClips = {},
        moments = {},
        weightRegConsts = {},
        initialRange = {}
    }

    local grad_last = {} -- add momentum to existing gradient
    if IDX_wordembed == 1 then
        -- Word embedding matrix is params_current[1]
        config_layers.lr_rates = {lr_wordembed}
        config_layers.gradientClips = {gradientClip_dummy}
        config_layers.weightClips = {weightClip_wordembed}
        config_layers.moments = {moments_dummy}
        config_layers.weightRegConsts = {weightRegConsts_dummy}
        config_layers.initialRange = {initialRange_dummy}
        for i = 2, #params_current do
            table.insert(config_layers.lr_rates, lr_other)
            table.insert(config_layers.moments, moments_dummy)
            table.insert(config_layers.gradientClips, gradientClip_dummy)
            table.insert(config_layers.weightClips, weightClip_other)
            table.insert(config_layers.weightRegConsts, weightRegConsts_dummy)
            table.insert(config_layers.initialRange, initialRange_dummy)    
        end

    else
        for i = 1, #params_current do
            table.insert(config_layers.lr_rates, lr_other)
            table.insert(config_layers.moments, moments_dummy)
            table.insert(config_layers.gradientClips, gradientClip_dummy)
            table.insert(config_layers.weightClips, weightClip_other)
            table.insert(config_layers.weightRegConsts, weightRegConsts_dummy)
            table.insert(config_layers.initialRange, initialRange_dummy)
        end
    end

    for i=1, #gparams_current do
        -- add momentum to existing gradient
        grad_last[i] = gparams_current[i]:clone()
        grad_last[i]:fill(0)
    end
    return config_layers, grad_last
end

---------------------------------------
---- data IO relevant functions--------
---------------------------------------
function existfile(filename)
    local f=io.open(filename,"r")
    if f~=nil then io.close(f) return true else return false end
end

--------------------------------
-- Helper function for img id --
--------------------------------
function load_filelist(fname)
    local data = file.read(fname)
    data = stringx.replace(data,'\n',' ')
    data = stringx.split(data)
    local imglist_ind = {}
    for i=1, #data do
        imglist_ind[i] = stringx.split(data[i],'.')[1]
    end
    return imglist_ind
end

----------------------
-- Build Vocabulary --
----------------------
function build_vocab(data, thresh, IDX_singleline, IDX_includeEnd)
    if IDX_singleline == 1 then
        data = stringx.split(data,'\n')
    else
        data = stringx.replace(data,'\n', ' ')
        data = stringx.split(data)
    end
    local countWord = {}
    for i=1, #data do
        if countWord[data[i]] == nil then
            countWord[data[i]] = 1
        else
            countWord[data[i]] = countWord[data[i]] + 1
        end
    end
    local vocab_map_ = {}
    local ivocab_map_ = {}
    local vocab_idx = 0
    if IDX_includeEnd==1 then
        vocab_idx = 1
        vocab_map_['NA'] = 1
        ivocab_map_[1] = 'NA'
    end

    for i=1, #data do
        if vocab_map_[data[i]]==nil then
            if countWord[data[i]]>=thresh then
                vocab_idx = vocab_idx+1
                vocab_map_[data[i]] = vocab_idx
                ivocab_map_[vocab_idx] = data[i]
            else
                vocab_map_[data[i]] = vocab_map_['NA']
            end
        end 
    end
    vocab_map_['END'] = -1
    return vocab_map_, ivocab_map_, vocab_idx
end

------------------------------------------------
-- Load img features and processed text files --
---------- Target and Other questions ----------
------------------------------------------------
function load_visualqadataset(opt, dataType, manager_vocab)
    local path_dataset = 'vqa_data_share/'

    ---------------
    -- All files --
    ---------------
    local prefix = 'coco_' .. dataType
    local filename_question = paths.concat(path_dataset, prefix .. '_question.txt')
    local filename_answer = paths.concat(path_dataset, prefix .. '_answer.txt')
    local filename_imglist = paths.concat(path_dataset, prefix .. '_imglist.txt')
    local filename_allanswer = paths.concat(path_dataset, prefix .. '_allanswer.txt')
    local filename_choice = paths.concat(path_dataset, prefix .. '_choice.txt')
    local filename_question_type = paths.concat(path_dataset, prefix .. '_question_type.txt')
    local filename_answer_type = paths.concat(path_dataset, prefix .. '_answer_type.txt')
    local filename_questionID = paths.concat(path_dataset, prefix .. '_questionID.txt')

    ------------------------------
    -- Target question filename --
    ------------------------------
    local target_prefix = 'target_coco_' .. dataType
    local target_filename_question = paths.concat(path_dataset, target_prefix .. '_question.txt')
    local target_filename_answer = paths.concat(path_dataset, target_prefix .. '_answer.txt')
    local target_filename_imglist = paths.concat(path_dataset, target_prefix .. '_imglist.txt')
    local target_filename_choice = paths.concat(path_dataset, target_prefix .. '_choice.txt')
    local target_filename_question_type = paths.concat(path_dataset, target_prefix .. '_question_type.txt')
    local target_filename_answer_type = paths.concat(path_dataset, target_prefix .. '_answer_type.txt')
    local target_filename_questionID = paths.concat(path_dataset, target_prefix .. '_questionID.txt')

    --------------------------
    -- Other question files --
    --------------------------
    local other_prefix = 'other_coco_' .. dataType
    local other_filename_question = paths.concat(path_dataset, other_prefix .. '_question.txt')
    local other_filename_answer = paths.concat(path_dataset, other_prefix .. '_answer.txt')
    local other_filename_imglist = paths.concat(path_dataset, other_prefix .. '_imglist.txt')
    local other_filename_choice = paths.concat(path_dataset, other_prefix .. '_choice.txt')
    local other_filename_question_type = paths.concat(path_dataset, other_prefix .. '_question_type.txt')
    local other_filename_answer_type = paths.concat(path_dataset, other_prefix .. '_answer_type.txt')
    local other_filename_questionID = paths.concat(path_dataset, other_prefix .. '_questionID.txt')

    --------------------
    -- Load all files --
    --------------------
    if existfile(filename_allanswer) then
        data_allanswer = file.read(filename_allanswer)
        data_allanswer = stringx.split(data_allanswer,'\n')
    else
        print(filename_allanswer .. " does not exist")
        os.exit()        
    end

    if existfile(filename_choice) then
        data_choice = file.read(filename_choice)
        data_choice = stringx.split(data_choice, '\n')
    else
        print(filename_choice .. " does not exist") 
        os.exit()   
    end

    if existfile(filename_question_type) then
        data_question_type = file.read(filename_question_type)
        data_question_type = stringx.split(data_question_type,'\n')
    else
        print(filename_question_type .. " does not exist")
        os.exit()    
    end

    if existfile(filename_answer_type) then
        data_answer_type = file.read(filename_answer_type)
        data_answer_type = stringx.split(data_answer_type, '\n')
    else
        print(filename_answer_type .. " does not exist")
        os.exit()    
    end

    if existfile(filename_answer) then
        data_answer = file.read(filename_answer)
        data_answer_split = stringx.split(data_answer,'\n')
    else 
        print(filename_answer .. " does not exist")
        os.exit()
    end

    if existfile(filename_question) then
        data_question = file.read(filename_question)
        data_question_split = stringx.split(data_question,'\n')
    else
        print(filename_question .. " does not exist")
        os.exit()    
    end

    -- only for TEST files
    if  existfile(filename_questionID) then
        data_questionID = file.read(filename_questionID)
        data_questionID = stringx.split(data_questionID,'\n')
    else
        print(filename_questionID .. " does not exist; Ignore for training and validation sets")   
    end

    ----------------------------
    -- Load target data files --
    ----------------------------
    if existfile(target_filename_choice) then
        target_data_choice = file.read(target_filename_choice)
        target_data_choice = stringx.split(target_data_choice, '\n')
    else
        print(target_filename_choice .. " does not exist") 
        os.exit()   
    end

    if existfile(target_filename_question_type) then
        target_data_question_type = file.read(target_filename_question_type)
        target_data_question_type = stringx.split(target_data_question_type,'\n')
    else
        print(target_filename_question_type .. " does not exist")
        os.exit()    
    end

    if existfile(target_filename_answer_type) then
        target_data_answer_type = file.read(target_filename_answer_type)
        target_data_answer_type = stringx.split(target_data_answer_type, '\n')
    else
        print(target_filename_answer_type .. " does not exist")
        os.exit()    
    end

    if existfile(target_filename_answer) then
        target_data_answer = file.read(target_filename_answer)
        target_data_answer_split = stringx.split(target_data_answer,'\n')
    else 
        print(target_filename_answer .. " does not exist")
        os.exit()
    end

    if existfile(filename_question) then
        target_data_question = file.read(target_filename_question)
        target_data_question_split = stringx.split(target_data_question,'\n')
    else
        print(target_filename_question .. " does not exist")
        os.exit()    
    end

    -- only for TEST files
    if  existfile(target_filename_questionID) then
        target_data_questionID = file.read(target_filename_questionID)
        target_data_questionID = stringx.split(target_data_questionID,'\n')
    else
        print(target_filename_questionID .. " does not exist; Ignore for training and validation sets")   
    end

    ---------------------------
    -- Load other data files --
    ---------------------------
    if existfile(other_filename_choice) then
        other_data_choice = file.read(other_filename_choice)
        other_data_choice = stringx.split(other_data_choice, '\n')
    else
        print(other_filename_choice .. " does not exist") 
        os.exit()   
    end

    if existfile(other_filename_question_type) then
        other_data_question_type = file.read(other_filename_question_type)
        other_data_question_type = stringx.split(other_data_question_type,'\n')
    else
        print(other_filename_question_type .. " does not exist")
        os.exit()    
    end

    if existfile(other_filename_answer_type) then
        other_data_answer_type = file.read(other_filename_answer_type)
        other_data_answer_type = stringx.split(other_data_answer_type, '\n')
    else
        print(other_filename_answer_type .. " does not exist")
        os.exit()    
    end

    if existfile(other_filename_answer) then
        other_data_answer = file.read(other_filename_answer)
        other_data_answer_split = stringx.split(other_data_answer,'\n')
    else 
        print(other_filename_answer .. " does not exist")
        os.exit()
    end

    if existfile(other_filename_question) then
        other_data_question = file.read(other_filename_question)
        other_data_question_split = stringx.split(other_data_question,'\n')
    else
        print(other_filename_question .. " does not exist")
        os.exit()    
    end

    -- only for TEST files
    if  existfile(other_filename_questionID) then
        other_data_questionID = file.read(other_filename_questionID)
        other_data_questionID = stringx.split(other_data_questionID,'\n')
    else
        print(other_filename_questionID .. " does not exist; Ignore for training and validation sets")   
    end

    ----------------------
    -- Build Vocabulary --
    -- Ques & Ans sets ---
    ----------------------
    local manager_vocab_ = {}

    if manager_vocab == nil then
        local vocab_map_answer, ivocab_map_answer, nvocab_answer = build_vocab(data_answer, opt.thresh_answerword, 1, 0)
        local vocab_map_question, ivocab_map_question, nvocab_question = build_vocab(data_question,opt.thresh_questionword, 0, 1)
        -- print('Vocabulary for Questions = ' .. nvocab_question.. ', Vocabulary for Answers = ' .. nvocab_answer)
        manager_vocab_ = {vocab_map_answer=vocab_map_answer, ivocab_map_answer=ivocab_map_answer, vocab_map_question=vocab_map_question, ivocab_map_question=ivocab_map_question, nvocab_answer=nvocab_answer, nvocab_question=nvocab_question}
    else
        -- TRAINING
        -- local vocab_map_answer, ivocab_map_answer, nvocab_answer = build_vocab(data_answer, opt.thresh_answerword, 1, 0)
        -- local vocab_map_question, ivocab_map_question, nvocab_question = build_vocab(data_question,opt.thresh_questionword, 0, 1)
        -- local vocab_map_extra_question, ivocab_map_extra_question, nvocab_extra_question = build_vocab(data_extra_question,opt.thresh_questionword, 0, 1)
        -- local vocab_map_question, ivocab_map_question, nvocab_question = build_vocab(data_total_question,opt.thresh_questionword, 0, 1)
        -- manager_vocab_ = {vocab_map_answer=vocab_map_answer, ivocab_map_answer=ivocab_map_answer, vocab_map_question=vocab_map_question, ivocab_map_question=ivocab_map_question, nvocab_answer=nvocab_answer, nvocab_question=nvocab_question}
        
        -- TESTING
        manager_vocab_ = manager_vocab
    end

    ----------------
    -- Image list --
    ----------------
    local imglist = load_filelist(filename_imglist)
    local other_imglist = load_filelist(other_filename_imglist) -- same for other
    local target_imglist = load_filelist(target_filename_imglist) -- same for target
    local nSample = #imglist
    if nSample > #data_question_split then
        nSample = #data_question_split
    end

    -------------
    -- Answers --
    -------------
    local x_answer = torch.zeros(nSample):fill(-1)
    if opt.multipleanswer == 1 then
        x_answer = torch.zeros(nSample, 10)
    end
    local x_answer_num = torch.zeros(nSample)

    ---------------------------------------------
    -- Generate BOW rule for entire vocabulary --
    ---------------------------------------------
    local x_question = torch.zeros(nSample, opt.seq_length)
    for i = 1, nSample do -- 68k
        local words = stringx.split(data_question_split[i])
        if existfile(filename_answer) then
            local answer = data_answer_split[i]
            if manager_vocab_.vocab_map_answer[answer] == nil then
                x_answer[i] = 1
            else
                x_answer[i] = manager_vocab_.vocab_map_answer[answer]
            end
        end
        for j = 1, opt.seq_length do -- 50
            if j <= #words then
                if manager_vocab_.vocab_map_question[words[j]] == nil then
                    x_question[{i, j}] = 1
                else
                    x_question[{i, j}] = manager_vocab_.vocab_map_question[words[j]]
                end
            else
        if manager_vocab_.vocab_map_question['END'] == nil then
                    x_question[{i, j}] = 1
                else
                    x_question[{i, j}] = manager_vocab_.vocab_map_question['END']
                end
            end
        end
    end
    
    --------------------
    -- Image features --
    --------------------
    local featureMap = {}
    local featName = 'googlenetFCdense' -- alexnet -- resnet

    ------------------------------------
    -- Rule for combining train & val --
    -------- Performing test -----------
    --- Select which dataset to use ----
    ------------------------------------
    local loading_spec = {
        trainval2014 = { train = true, val = true, test = false },
        trainval2014_train = { train = true, val = true, test = false },
        trainval2014_val = { train = false, val = true, test = false },
        train2014 = { train = true, val = false, test = false },
        val2014 = { train = false, val = true, test = false },
        test2015 = { train = false, val = false, test = true }
    }

    loading_spec['test-dev2015'] = { train = false, val = false, test = true }
    
    local feature_prefixSet = {
        train = paths.concat(path_dataset, 'coco_train2014_' .. featName), 
        val = paths.concat(path_dataset, 'coco_val2014_' .. featName),
        test = paths.concat(path_dataset,'coco_test2015_' .. featName)
    }

    for k, feature_prefix in pairs(feature_prefixSet) do
        if loading_spec[dataType][k] then
            local feature_imglist = torch.load(feature_prefix  ..'_imglist.dat')
            local featureSet = torch.load(feature_prefix ..'_feat.dat')
            for i = 1, #feature_imglist do
                local feat_in = torch.squeeze(featureSet[i])
                featureMap[feature_imglist[i]] = feat_in
            end
        end
    end

    collectgarbage()

    local _state = {
        x_question = x_question, 
        x_answer = x_answer, 
        x_answer_num = x_answer_num, 
        featureMap = featureMap, 
        data_question = data_question_split,
        data_answer = data_answer_split, 
        imglist = imglist, 
        path_imglist = path_imglist, 
        data_allanswer = data_allanswer, 
        data_choice = data_choice, 
        data_question_type = data_question_type, 
        data_answer_type = data_answer_type, 
        data_questionID = data_questionID

    }
    return _state, manager_vocab_
end

------------------------
-- Save trained model --
------------------------
function save_model(opt, manager_vocab, context, path)
    print('saving model ' .. path)
    local d = {}
    d.paramx = context.paramx:float()
    d.manager_vocab = manager_vocab
    d.stat = stat
    d.config_layers = config_layers
    d.opt = opt
    torch.save(path, d)
end

---------------------------
-- Get existing BOW rule --
-- Input: Word index ------
-- Output: BOW vector -----
---------------------------
function bagofword(manager_vocab, x_seq)
    local outputVector = torch.zeros(manager_vocab.nvocab_question)
    for i= 1, x_seq:size(1) do
        if x_seq[i] ~= manager_vocab.vocab_map_question['END'] then    
            outputVector[x_seq[i]] = 1
        else
            break
        end
    end
    return outputVector
end

------------------------------------
---------- Generate table ----------
-------- Input = key,value ---------
--- Output = table[key] = value ----
------------------------------------
function add_count(t, ...)
    local args = { ... }
    local i = 1
    while i < #args do
        local k = args[i]
        local v = args[i + 1]
        if t[k] == nil then 
            t[k] = { v, 1 } 
        else
            t[k][1] = t[k][1] + v
            t[k][2] = t[k][2] + 1
        end
        i = i + 2
    end
end

-------------------------------------
------ Compute Total Accuracy -------
-- Input: Per question type result --
--------- Output: Accuracy ----------
------------------------------------- 
function compute_accuracy(t)
    local res = { }
    for k, v in pairs(t) do
        res[k] = v[1] / v[2]
    end
    return res
end

-------------
-- Testing --
-------------
function evaluate_answer(state, manager_vocab, pred_answer, prob_answer, selectIDX)
    selectIDX = selectIDX or torch.range(1, state.x_answer:size(1))
    local pred_answer_word = {}
    local gt_answer_word = state.data_answer
    local gt_allanswer = state.data_allanswer

    local perfs = { } 
    local count_question_type = {}
    local count_answer_type = {}

    for sampleID = 1, selectIDX:size(1) do
        local i = selectIDX[sampleID]

        -- Correct answer
        if manager_vocab.ivocab_map_answer[pred_answer[i]]== gt_answer_word[i] then
            add_count(perfs, "most_freq", 1)
        else
            add_count(perfs, "most_freq",0)
        end

        -- Standard criteria (min(#correct match/3, 1))
        local question_type = state.data_question_type[i]
        local answer_type = state.data_answer_type[i]

        -- Multiple choice
        local choices = stringx.split(state.data_choice[i], ',')
        local score_choices = torch.zeros(#choices):fill(-1000000)
        for j = 1, #choices do
            local IDX_pred = manager_vocab.vocab_map_answer[choices[j]]
            if IDX_pred ~= nil then
                local score = prob_answer[{i, IDX_pred}]
                if score ~= nil then
                    score_choices[j] = score
                end
            end
        end
        local val_max, IDX_max = torch.max(score_choices, 1)
        local word_pred_answer_multiple = choices[IDX_max[1]]
        local word_pred_answer_openend = manager_vocab.ivocab_map_answer[pred_answer[i]]

        -- Compare the predicted answer with all ground truth answers from humans.
        if gt_allanswer then
            local answers = stringx.split(gt_allanswer[i], ',')
            -- The number of answers matched with human answers.
            local count_curr_openend = 0
            local count_curr_multiple = 0
            for j = 1, #answers do
                count_curr_openend = count_curr_openend + (word_pred_answer_openend == answers[j] and 1 or 0)
                count_curr_multiple = count_curr_multiple + (word_pred_answer_multiple == answers[j] and 1 or 0)
            end

            local increment = math.min(count_curr_openend * 1.0/3, 1.0)
            add_count(perfs, "openend_overall", increment, 
                             "openend_q_" .. question_type, increment, 
                             "openend_a_" .. answer_type, increment)

            increment = math.min(count_curr_multiple * 1.0/3, 1.0)
            add_count(perfs, "multiple_overall", increment, 
                             "multiple_q_" .. question_type, increment, 
                             "multiple_a_" .. answer_type, increment)
        end
    end

    -- Compute accuracy
    return compute_accuracy(perfs)
end

-----------------------
-- Write predictions --
-----------------------
function outputJSONanswer(state, manager_vocab, prob, file_json, choice)
    local f_json = io.open(file_json,'w')
    f_json:write('[')
    
    for i = 1, prob:size(1) do
        local choices = stringx.split(state.data_choice[i], ',')
        local score_choices = torch.zeros(#choices):fill(-1000000)
        for j=1, #choices do
            local IDX_pred = manager_vocab.vocab_map_answer[choices[j]]
            if IDX_pred ~= nil then
                local score = prob[{i, IDX_pred}]
                if score ~= nil then
                    score_choices[j] = score
                end
            end
        end
        local val_max,IDX_max = torch.max(score_choices,1)
        local val_max_open,IDX_max_open = torch.max(prob[i],1)
        local word_pred_answer_multiple = choices[IDX_max[1]]
        local word_pred_answer_openend = manager_vocab.ivocab_map_answer[IDX_max_open[1]]
        local answer_pred = word_pred_answer_openend
        -- choice is 1 for multiple choice questions
        if choice == 1 then
            answer_pred = word_pred_answer_multiple
        end
        local questionID = state.data_questionID[i]
        f_json:write('{"answer": "' .. answer_pred .. '","question_id": ' .. questionID .. '}')
        if i< prob:size(1) then
            f_json:write(',')
        end
    end
    f_json:write(']')
    f_json:close()

end

------------------------
-- Save local context --
------------------------
function train_epoch(opt, state, manager_vocab, context, updateIDX)
    local model = context.model
    local criterion = context.criterion
    local paramx = context.paramx
    local paramdx = context.paramdx
    local params_current = context.params_current
    local gparams_current = context.gparams_current
    local config_layers = context.config_layers
    -- add momentum to existing gradient
    local grad_last = context.grad_last
 
    local loss = 0.0
    local N = math.ceil(state.x_question:size(1) / opt.batchsize)
    local prob_answer = torch.zeros(state.x_question:size(1), manager_vocab.nvocab_answer)
    local pred_answer = torch.zeros(state.x_question:size(1))
    local target = torch.zeros(opt.batchsize)

    local featBatch_visual = torch.zeros(opt.batchsize, opt.vdim)
    local featBatch_word = torch.zeros(opt.batchsize, manager_vocab.nvocab_question)
    local word_idx = torch.zeros(opt.batchsize, opt.seq_length)

    local IDXset_batch = torch.zeros(opt.batchsize)
    local nSample_batch = 0
    local count_batch = 0
    local nBatch = 0

    local randIDX = torch.randperm(state.x_question:size(1))
    for iii = 1, state.x_question:size(1) do
        local i = randIDX[iii]
        local first_answer = -1
        if updateIDX~='test' then
            first_answer = state.x_answer[i]
        end
        if first_answer == -1 and updateIDX == 'train' then
            ----------------------------------
            ----------- Do nothing ----------- 
            -- Skip samples with NA answers -- 
            ----------------------------------
        else
            nSample_batch = nSample_batch + 1
            IDXset_batch[nSample_batch] = i
            if updateIDX ~= 'test' then
                target[nSample_batch] = state.x_answer[i]
            end
            -- print(state.imglist)

            local filename = state.imglist[i]
            --[[
            -- necessary for 'test-dev2015'
            -- filename **must** be the complete filename and not the image id
            for k, v in pairs(state.featureMap) do
                if filename == tonumber(k:split('_')[3]) then
                  filename = k
                end
            end
            ]]
            
            local feat_visual = state.featureMap[filename]:clone()   
            ---------------------------
            -- Get existing BOW rule --
            -- Input: Word index ------
            -- Output: BOW vector -----
            ---------------------------
            local feat_word = bagofword(manager_vocab, state.x_question[i])
            word_idx[nSample_batch] = state.x_question[i]
            featBatch_word[nSample_batch] = feat_word:clone()
            featBatch_visual[nSample_batch] = feat_visual:clone()          
            while i == state.x_question:size(1) and nSample_batch < opt.batchsize do
                -------------
                -- Padding -- 
                -------------
                nSample_batch = nSample_batch+1
                IDXset_batch[nSample_batch] = i
                target[nSample_batch] = first_answer
                featBatch_visual[nSample_batch] = feat_visual:clone()
                featBatch_word[nSample_batch] = feat_word:clone()
                word_idx[nSample_batch] = state.x_question[i]
            end 
            if nSample_batch == opt.batchsize then                
                nBatch = nBatch+1
                word_idx = word_idx:cuda()
                nSample_batch = 0
                target = target:cuda()
                featBatch_word = featBatch_word:cuda()
                featBatch_visual = featBatch_visual:cuda()
                ------------------
                -- Forward pass --
                ------------------

                --switch between the baselines and the memn2n
                input = {featBatch_word, featBatch_visual}

                local output = model:forward(input)
                local err = criterion:forward(output, target)
                local prob_batch = output:float()

                loss = loss + err
                for j = 1, opt.batchsize do
                    prob_answer[IDXset_batch[j]] = prob_batch[j]
                end
                -------------------
                -- Backward pass --
                -------------------
                if updateIDX == 'train' then
                    model:zeroGradParameters()
                    local df = criterion:backward(output, target)
                    local df_model = model:backward(input, df)
                    ----------------------
                    -- Parameter update --
                    ----------------------
                    if opt.uniformLR ~= 1 then
                        for i=1, #params_current do
                            local gnorm = gparams_current[i]:norm()
                            if config_layers.gradientClips[i]>0 and gnorm > config_layers.gradientClips[i] then
                                gparams_current[i]:mul(config_layers.gradientClips[i]/gnorm)
                            end
                            -- add momentum to existing gradient
                            grad_last[i]:mul(config_layers.moments[i])
                            local tmp = torch.mul(gparams_current[i],-config_layers.lr_rates[i])
                            grad_last[i]:add(tmp)
                            params_current[i]:add(grad_last[i])
                            if config_layers.weightRegConsts[i]>0 then
                                local a = config_layers.lr_rates[i] * config_layers.weightRegConsts[i]
                                params_current[i]:mul(1-a)
                            end
                            local pnorm = params_current[i]:norm()
                            if config_layers.weightClips[i]>0 and pnorm > config_layers.weightClips[i] then
                                params_current[i]:mul(config_layers.weightClips[i]/pnorm)
                            end
                        end
                    else
                        local norm_dw = paramdx:norm()
                        if norm_dw > opt.max_gradientnorm then
                            local shrink_factor = opt.max_gradientnorm / norm_dw
                            paramdx:mul(shrink_factor)
                        end
                        paramx:add(g_paramdx:mul(-opt.lr))
                    end
 
                end

                ------------------
                -- End of batch --
                ------------------
                count_batch = count_batch+1
                if count_batch == 120 then
                    collectgarbage()
                    count_batch = 0
                end
            end
        end-- end of the pass sample with -1 answer IDX
    end
    ------------------
    -- End of epoch --
    ------------------
    local y_max, i_max = torch.max(prob_answer,2)
    i_max = torch.squeeze(i_max)
    pred_answer = i_max:clone() 
    if updateIDX~='test' then

        local gtAnswer = state.x_answer:clone()
        gtAnswer = gtAnswer:long()
        local correctNum = torch.sum(torch.eq(pred_answer, gtAnswer))
        acc = correctNum*1.0/pred_answer:size(1)
    else
        acc = -1
    end
    print(updateIDX ..': acc (mostFreq) = ' .. acc)
    local perfs = nil -- per question type result stored
    ---------------------------------
    -- Standard evalution criteria --
    ------ from Virgina Tech --------
    ---------------------------------
    if updateIDX ~= 'test' and state.data_allanswer ~= nil then
        perfs = evaluate_answer(state, manager_vocab, pred_answer, prob_answer)
        print(updateIDX .. ': acc.match mostfreq = ' .. perfs.most_freq)
        -- print(updateIDX .. ': acc.dataset (OpenEnd) = ' .. perfs.openend_overall)
        print(updateIDX .. ': acc.dataset (MultipleChoice) = ' .. perfs.multiple_overall)
        -- to see per question type result
        -- print(perfs)
    end
    print(updateIDX .. ' loss=' .. loss/nBatch)
    return pred_answer, prob_answer, perfs
end