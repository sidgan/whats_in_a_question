#!/usr/bin/env python

#####################################
# create lmdb for images and labels #
#####################################

import sys
sys.path.insert(0,"/nfs.yoda/sganju1/caffe/python")
# the folder where you have the python caffe. Note that this path is without the /caffe at the end.
import caffe
sys.path.insert(0,'/nfs.yoda/sganju1/caffe/python/caffe')
#Note that this has /caffe at the end
caffe.set_mode_gpu()
import lmdb
from PIL import Image

classifier_id = np.array([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0])

in_db = lmdb.open('classifier-id-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(imgids):
        im = np.array(classifier_id)
        im = np.reshape(im, (len(im), 1, 1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())

in_db.close()