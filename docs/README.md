## What’s in a Question: Using Visual Questions as a Form of Supervision

[Siddha Ganju](http://sidgan.me/siddhaganju), [Olga Russakovsky](http://www.cs.cmu.edu/~orussako/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg/)

<img src="https://www.cmu.edu/marcom/brand-standards/images/logos-colors-type/full-color-seal-min.png" width="425" height="300"/> <img src="http://cvpr2017.thecvf.com/images/CVPRLogo3.jpg" width="425"/>

![What’s in a Question: Using Visual Questions as a Form of Supervision](https://raw.githubusercontent.com/sidgan/whats_in_a_question/master/pullfig.png?token=ADb_B1zRklwR9tzKktfs29S4JvowmhaYks5Y4sRjwA%3D%3D)

### Abstract

Collecting fully annotated image datasets is challenging and expensive. Many types of weak supervision have been explored: weak manual annotations, web search results, temporal continuity, ambient sound, and others. We focus on one particular unexplored mode: visual questions that are asked about images. Our work is based on the key observation that the question itself provides useful information about the image (even without the answer being available). For instance, the question “what is the breed of the dog?” informs the computer that the animal in the scene is a dog and that there is only one dog present. We make three contributions: (1) we provide an extensive qualitative and quantitative analysis of the information contained in human visual questions, (2) we propose two simple but surprisingly effective modifications to the standard visual question answering models that allows it to make use of weak supervision in the form of unanswered questions associated with images, and (3) we demonstrate that a simple data augmentation strategy inspired by our insights results in a 7.1% improvement on the standard VQA benchmark.



---

### Citation

```
@inproceedings{GanjuCVPR17,
author = {Siddha Ganju and Olga Russakovsky and Abhinav Gupta},
title = {What's in a Question: Using Visual Questions as a Form of Supervision},
booktitle = {CVPR},
year = {2017}
}
```
---


### Links
+ [Paper]()
+ [CVPR'17 Poster]()
+ [Code]()
+ [Preprocessed data]()
+ [Models]()



There are three tasks described in the paper: 

### 1. Image Descriptions

We analyze whether the visual questions contain enough information to provide an accurate description of the image using the Seq2Seq model. See [Image Descriptions README](https://github.com/sidgan/cvpr2017/blob/master/image_descriptions/README.md) for detailed description for each file.

### 2. Object Classification

Visual questions can provide information about the object classes that are present in the image. E.g., asking “what color is the bus?” indicates the presence of a bus in the image. See [Object Classification README](https://github.com/sidgan/cvpr2017/blob/master/object_classification/README.md) for detailed description for each file.

#### Training/fine-tuning the image features (caffe)

Fine-tuning modifies only the last layer of a network to give the application-specific number of outputs. For fine-tuning we start with the parameters initially learnt on the ImageNet images, and then fine-tune with MS COCO images. All caffe related code for fine-tuning the models is present in the `caffe` directory. See [caffe README](https://github.com/sidgan/cvpr2017/blob/master/caffe/README.md) for detailed description for each file.

### 3. Visual Question Answering

Visual Question Answering is, given an image and a natural language question about the image, the task is to provide an accurate natural language answer. Visual questions focus on different areas of an image, including background details and underlying context. We utilize not just the target question, but also the unanswered questions about a particular image. See [Visual Question Answering README](https://github.com/sidgan/cvpr2017/blob/master/vqa/README.md) for detailed description for each file.


