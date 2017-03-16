#!/usr/bin/env th

require 'torch'
require 'optim'
require 'paths'
require 'xlua'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)

if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end

--------------
-- Training --
--------------
paths.dofile('train.lua')

-------------
-- Testing --
-------------
paths.dofile('test.lua')