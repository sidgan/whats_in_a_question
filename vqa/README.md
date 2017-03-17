
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

Download the processed data files for various combinations of the `target question` and `unanswered questions`. Folder names are according to the names of the `unanswered question` because the random target questions remain the same.

**Unanswered Questions**

As described in the [paper](), there are 2^3 combinations possible for `target question`-`unanswered question` pair. Consider an image `x` with a question `q`, a corresponding answer `a` and two additional unanswered questions `q_1` and `q_2`. For [iBOWIMG](https://github.com/metalbubble/VQAbaseline), the single training example corresponding to this image would be `(x, q, a)`. For **iBOWIMG-2x** there would be eight training examples, with `E = {null, q, q_1, q_2, [q,q_1], [q,q_2], [q_1,q_2], [q,q_1,q_2]}` making use of the extra information that is available about this image during training in the form of unanswered asked questions. 


|Training Example| `unanswered Question` Datafile link |
|--|-|
|`null`| For the `null` `unanswered question`, use the `null tensor` in Torch. |
| `q`| [Unanswered question is same as the target question (same link as 1 Randomly chosen target question)](https://cmu.box.com/s/ht8zjwik7n9atq36p0dcd7x3w87f5x8k) 
|`q_1`|[1 Randomly chosen unanswered question from the remaining 2 questions (hence, not the target question)](https://cmu.box.com/s/uqtwafj6agp5ilmib58faxxoi7yhfsj0) |
| `q_2`| [1 Randomly chosen unanswered question from the remaining 2 questions (hence, not the target question)-2](https://cmu.box.com/s/5akxvquu45q93z1ler8ow042lgux49nx) | 
|`[q,q_1]`| [Target question and 1 randomly chosen from the remaining 2 questions](https://cmu.box.com/s/zt37h3yf5c2pum2ua4gb1owpa48qpzjn) | 
| `[q,q_2]`| [Target question and 1 randomly chosen from the remaining 2 questions-2](https://cmu.box.com/s/hp6uzb858ka4cbinb389vqhw1d7xxppo) | 
| `[q_1,q_2]`| [unanswered 2 questions](https://cmu.box.com/s/7594vv6v5lm0548vh7f180svxqbge232) | 
| `[q,q_1,q_2]` | [All 3 questions (includes the target question)](https://cmu.box.com/s/tlcqdrr6fa2v3i2lrukj3wn3a8qkood5) |

**Target Question**

1 Randomly chosen target question: [Download here](https://cmu.box.com/s/ht8zjwik7n9atq36p0dcd7x3w87f5x8k)

**All training files**

Download the `all` training files [here](https://cmu.box.com/s/v148z8qik8xicj9pjdvsyhwt2irezo4o). These are needed to generate the vocabulary.

+ **Image Features**

Download the image features from AlexNet or GoogLeNet or ResNet in binary format and update the path. The zip files below provide the image features and the image list which maps the filename of the image to its corresponding features. Change the corresponding argument in the code to accept the image features.

|Link|Features from the _ model|
|-|-|
|[coco_val2014_googlenetFCdense_feat.dat](http://visualqa.csail.mit.edu/data_vqa_feat.zip) | GoogLeNet model |
|[coco_val2014_alexnet_feat.dat](https://cmu.box.com/s/1zjrnmv3nma4n1590gdmf6yx6gf0ungy) | AlexNet model |
|[coco_val2014_resnet_feat.dat](https://cmu.box.com/s/806eubldsk01qxoi1fgf1t6wknyctw5r) |  ResNet model |

After installing all the necessary components, run `th main.lua` and call the training components.


### Testing the VQA model

After installing all the necessary components, run `th main.lua` and call the testing components.

