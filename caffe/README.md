## caffe

All caffe related code for fine-tuning the models is present in this directory. Fine-tuning modifies only the last layer of a network to give the application-specific number of outputs. For fine-tuning we start with the parameters initially learnt on the ImageNet images, and then fine-tune with MS COCO images. The weights from the pretrained model are passed as an argument to the caffe train command which loads the pretrained weights according to the names of each layer. 

### Procedure

#### Data

Generate the LMDB format for images using `generate_lmdb.py`.

##### Fine Tune

`./build/tools/caffe train -solver solver.prototxt -weights bvlc_reference_caffenet.caffemodel`

Please note that all the paths are absolute.


We have to predict the 92 MS COCO (or the vocabulary words) classes instead of the ImageNet 1,000, so the following is changed in the last layer in the model. 

```
layer {
  name: "fc8_flickr"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8_flickr"
  # lr_mult is set to higher than for other layers, because this layer is starting from random while the others are already trained
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 92
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
```

The name of the last layer is changed from `fc8` to `fc8_flickr`. As there is no layer named `fc8_flickr` in the bvlc_reference_caffenet, this layer will begin training with random weights.

The overall learning rate `base_lr` is decreased in the `solver.prototxt` and `lr_mult` is set to a value higher than the rest, as this layer is starting from random while the others have pretrained weights. This example provided is for AlexNet. The finetuning of GoogLeNet and ResNet follow suit.

File descriptions:

1. [monitor/extract_seconds.py](https://github.com/sidgan/cvpr2017/blob/master/caffe/monitor/extract_seconds.py): Visualization of training progress
2. [monitor/parse_log.sh](https://github.com/sidgan/cvpr2017/blob/master/caffe/monitor/parse_log.sh): Visualization of training progress
3. [monitor/progress_plot.py](https://github.com/sidgan/cvpr2017/blob/master/caffe/monitor/progress_plot.py): Visualization of training progress
4. [alexnet_classify.py](https://github.com/sidgan/cvpr2017/blob/master/caffe/alexnet_classify.py): Classify the test/validation images using the trained model. Will produce a python pickle file. 
5. [generate_lmdb.py](https://github.com/sidgan/cvpr2017/blob/master/caffe/generate_lmdb.py): Generate image-label dataset in LMDB format. 
6. [alexnet/deploy.prototxt](https://github.com/sidgan/cvpr2017/blob/master/caffe/alexnet/deploy.prototxt): Finetuning and Training Caffe models
7. [alexnet/solver.prototxt](https://github.com/sidgan/cvpr2017/blob/master/caffe/alexnet/solver.prototxt): Finetuning and Training Caffe models
8. [alexnet/train_val.prototxt](https://github.com/sidgan/cvpr2017/blob/master/caffe/alexnet/train_val.prototxt): Finetuning and Training Caffe models

See [Fine-tuning CaffeNet for Style Recognition on “Flickr Style” Data](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) for more information on finetuning and [Brewing ImageNet](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html) for more information on Caffe Training.