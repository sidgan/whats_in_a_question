local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Visual Question Answering')
   cmd:text()
   cmd:text('Options:')

   --------------------
   -- Model settings --
   --------------------
   cmd:option('--savepath', 'models', 'Path to save the trained model')
   
   --------------------
   -- Image Features --
   --------------------
   cmd:option('--vfeat', 'alexnet', 'Image Features') --googlenetFCdense --resnet
   cmd:option('--vdim', 1024, 'Image Feature Dimensions')

   -------------------------
   -- Data pre-processing --
   -------------------------
   cmd:option('--thresh_questionword',6, 'Word Frequencies for questions')
   cmd:option('--thresh_answerword', 3, 'Word Frequencies for answers')
   cmd:option('--batchsize', 100)
   cmd:option('--seq_length', 50)

   ------------------------------
   -- Learning rate and epochs --
   ------------------------------
   -- LR(word embedding layer) should be **much** higher than the LR(softmax layer) to learn a good word embedding
   cmd:option('--uniformLR', 0, 'Set uniform learning rate for learning all the parameters')
   cmd:option('--epochs', 100)
   cmd:option('--nepoch_lr', 100)
   cmd:option('--decay', 1.2)
   cmd:option('--embed_word', 1024,'Word Embedding')  
   cmd:option('--maxgradnorm', 20)
   cmd:option('--maxweightnorm', 2000)
   cmd:option('--lr_wordembed', 0.8)
   cmd:option('--lr_other', 0.01)
   -- weight clipping is important for better performance
   cmd:option('--weightClip_wordembed', 1500)
   cmd:option('--weightClip_other', 20)

   ---------------------------
   -- Cuda related and seed --
   ---------------------------
   cmd:option('-manualSeed', 43, 'Manually set RNG seed')
   cmd:option('-cuda', true, 'Use cuda.')
   cmd:option('-device', 1, 'Cuda device to use.')
   -- cutorch.setDevice(getFreeGPU())
   cmd:option('-nGPU',   2,  'Number of GPUs to use by default')
   cmd:option('-cudnn', true, 'Convert the model to cudnn.')
   cmd:option('-cudnn_bench', false, 'Run cudnn to choose fastest option. Increase memory usage')
   cmd:text()

   -- generate local options and return them 
   local opt = cmd:parse(arg or {})
   return opt
end

return M