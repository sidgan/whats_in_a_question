
## Training the VQA model (torch)

### Installation and set up

The code requires [Torch](http://torch.ch/docs/getting-started.html)

```
bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
$ luarocks install nn
$ luarocks install nngraph 
$ luarocks install image 
$ luarocks install cutorch
$ luarocks install cunn
```
Also install `fblualib`. [Installation for fblualib](https://github.com/facebook/fblualib).

This repo is over-riding some of the functions of the nn.LinearNB package and this may throw errors like [this](https://github.com/kaishengtai/torch-ntm/issues/8).

### Files

+ **Trained model**

Download the trained [iBOWIMG-2x model](https://cmu.box.com/s/84ujxt8pppazu6yof7xiwi2qkdiywjgs) and update the appropriate paths.

+ **Preprocessed text data**

Download the `all` training files [here](https://cmu.box.com/s/v148z8qik8xicj9pjdvsyhwt2irezo4o).

Download the processed data files for various combinations of the `target question` and `other questions`. Folder names are according to the names of the `other question` because the random target questions remain the same.

1 Randomly chosen target question: [Download here](https://cmu.box.com/s/ht8zjwik7n9atq36p0dcd7x3w87f5x8k) 

There are 2^3 combinations possible. For the `null` `other question`, use the `null tensor` in Torch.

| `Other Question` Datafiles |
|--|
| [1 Randomly chosen other question from the remaining 2 questions (hence, not the target question)](https://cmu.box.com/s/uqtwafj6agp5ilmib58faxxoi7yhfsj0) |
| [Other question is same as the target question (same link as 1 Randomly chosen target question)]()  |
| [1 Randomly chosen other question (could be same as target)](https://cmu.box.com/s/5akxvquu45q93z1ler8ow042lgux49nx) | 
| [Other 2 questions](https://cmu.box.com/s/7594vv6v5lm0548vh7f180svxqbge232) | 
| [Target question and 1 randomly chosen from the remaining 2 questions](https://cmu.box.com/s/zt37h3yf5c2pum2ua4gb1owpa48qpzjn) | 
| [Target question and 1 randomly chosen from the remaining 2 questions](https://cmu.box.com/s/zt37h3yf5c2pum2ua4gb1owpa48qpzjn) | 
| [All 3 questions (includes the target question )](https://cmu.box.com/s/tlcqdrr6fa2v3i2lrukj3wn3a8qkood5) |


+ **Image Features**

Download the image features from AlexNet or GoogLeNet or ResNet in binary format and update the path. The zip files below provide the image features and the image list which maps the filename of the image to its corresponding features. Change the corresponding argument in the code to accept the image features.

|Link|Feature|
|-|-|
|[coco_val2014_googlenetFCdense_feat.dat](http://visualqa.csail.mit.edu/data_vqa_feat.zip) | GoogLeNet model |
|[coco_val2014_alexnet_feat.dat]() | AlexNet model |
|[coco_val2014_resnet_feat.dat]() |  ResNet model |

After installing all the necessary components, run `th main.lua` and call the training components.


### Testing the VQA model

After installing all the necessary components, run `th main.lua` and call the testing components.

