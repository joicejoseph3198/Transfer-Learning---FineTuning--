# Using Transfer Learning for fine-tuning

> Using transfer learning to fine tune a CNN-VGG16 for classification of Oxford's Flowers-17 dataset using ImageNet weights 
---

### Table of Contents

- [Description](#description)
- [Working](#working)
- [Results](#results)


---

## Description

Fine-tuning requires us to perform “network surgery”. First, we cut off the final set of fully-connected layers (i.e., the “head” of the network) from a pre-trained Convolutional Neural Network, such as VGG, ResNet, or Inception. We then replace the head with a new set of fully-connected layers with random initializations. From there all layers below the head are frozen so their weights cannot be updated (i.e., the backward pass in backpropagation does not reach them). We then train the network using a very small learning rate so the new set of FC layers can start to learn patterns from the previously learned CONV layers earlier in the network. Optionally, we may unfreeze the rest of the network and continue training. Applying fine-tuning allows us to apply pre-trained networks to recognize classes that they were not originally trained on; furthermore, this method can lead to higher accuracy than you would normally achieve.

Applying fine-tuning is an extremely powerful technique as we do not have to train an entire network from scratch. Instead, we can leverage pre-existing network architectures, such as state-of-the-art models trained on the ImageNet dataset which consist of a rich, discriminative set of filters. Using these filters, we can “jump start” our learning, allowing us to perform network surgery, which ultimately leads to a higher accuracy transfer learning model with less effort than training from scratch.


[Back To The Top](#read-me-template)

---

## Working

#### Installation
The requirements document provided in the repository has the list of all the packages that are needed for implementing this project.

```html
    pip install -r requirements.txt 
```
* Choose a pretrained network (VGG-16 with ImageNet weights in present case).
* inspect_model.py : To examine the architecture and implementation of the pre-trained CNN. We need to know the layer name and index of every layer in a given deep learning model. This information will be required to “freeze” and “unfreeze” certain layers in a pre-trained CNN.

```html
    python inspect_model.py  
```
* fcheadnet. py :  To define our own fully-connected head of the network. I have defined the build method responsible for constructing the actual network architecture. This method requires three parameters: the baseModel (the body of the network), the total number of classes in our dataset, and finally D, the number of nodes in the fully-connected layer.
* This new FC head must have the final output nodes as equal to number of classes in the new dataset (flowers17 dataset has 17 classes, the architecture of the FC head I use in the present case is FullyConnected(256 nodes)+ReLU => DropOut(0.5) => FullyConnected(17 nodes)+SoftMax). 
* finetune_flowers17.py : To apply fine-tuning from start to finish. We’ll require two command line arguments for our script, --dataset, the path to the input directory containing the Flowers-17 dataset, and --model, the path to our output serialized weights after training.
    * We make use of certain separately defined preprocessing scripts inside this, namely ImageToArrayPreprocessor, AspectAwarePreprocessor, SimpleDatasetLoader.
    * We also perform data augumentation using ImageDataGenerator
    * Next we load the VGG16 architecture from disk using the supplied, pre-trained ImageNet weights. We purposely leave off the head of VGG16 as we’ll be replacing it with our own FCHeadNet. We also want to explicitly define the input_tensor to be 224x224x3 pixels.
    * We construct a new model using the body of VGG16 (baseModel.input) as the input and the headModel as the output.
    * We freeze the weights in the body so they are not updated during the backpropagation phase, accomplished by setting the .trainable parameters of the layers to False.
    * After connecting the head to the body and freezing the layers in the body, we can warmup the new head of the network, by letting it train for a few epochs.
    * After FC layers have been partly trained and initialized, we unfreeze some of the CONV layers in the body and make them trainable. And let the network train for 100 epochs.
    * We obtain an accuracy of nearly 95%.

```html
    python finetune_flowers17.py --dataset datasets\flowers17
    --model flowers17.model 
```  

## Results

Accuracy after the warm-up phase of FC head of the network

![Project Image1](https://raw.githubusercontent.com/joicejoseph3198/Images/main/finetuneflowers1.png)

Accuracy after the final phase of the entire network
![Project Image1](https://raw.githubusercontent.com/joicejoseph3198/Images/main/finetuneflowers2.png)

[Back To The Top](#read-me-template)




