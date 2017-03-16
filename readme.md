## What’s in a Question: Using Visual Questions as a Form of Supervision

This is the code for the CVPR'17 spotlight paper, [**What’s in a Question: Using Visual Questions as a Form of Supervision**](). 

The trained models attain the following scores on the test-dev of the [MS COCO VQA v1.0 dataset](http://www.visualqa.org/). 

|Model Name| Overall| Other |Number |Yes/No|
|-|-|-|-|-|
|iBOWIMG-2x  | 62.80  | 53.11  |37.94 | 80.72|

There are three tasks described in the paper: 

### 1. Image Descriptions

See [Image Descriptions Readme](https://github.com/sidgan/cvpr2017/blob/master/image_descriptions/readme.md)

### 2. Object Classification

See [Object Classification Readme](https://github.com/sidgan/cvpr2017/blob/master/object_classification/readme.md)

#### Training/fine-tuning the image features (caffe)

All caffe related code for fine-tuning the models is present in this directory. See [caffe readme](https://github.com/sidgan/cvpr2017/blob/master/caffe/readme.md) for detailed description for each file.

### 3. Visual Question Answering

See [VQA Readme](https://github.com/sidgan/cvpr2017/blob/master/vqa/readme.md)

### Acknowldegements

This code is based on [Simple Baseline for Visual Question Answering](https://arxiv.org/pdf/1512.02167.pdf) by Bolei Zhou and Yuandong Tian.

Please cite us if you use our code:

```
@inproceedings{GanjuCVPR17,
author = {Siddha Ganju and Olga Russakovsky and Abhinav Gupta},
title = {What's in a Question: Using Visual Questions as a Form of Supervision},
booktitle = {CVPR},
year = {2017}
}
```

