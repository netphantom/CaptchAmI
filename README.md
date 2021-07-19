# CaptchAmI - Stars and Numbers Captcha Solver

## What is CaptchAmI Captcha Solver
I made this project to show how it is possible to use AI + Computer Vision to solve Captcha.
In particular, I had to deal with images containing two captcha types: stars and numbers.
The main objective of this tool is to count the number of stars, or solve a small operation on numbers.


## Components
The project has 2 components: 

1) The Offline training component, to allow the training of the neural network

2) A webserver to use the pre-trained neural networks and get the results of the classification

## Computer Vision elaboration
### Stars elaboration
This elaboration relies only on Computer Vision techniques. 
To recognize how many stars are present, the image is preprocessed: it is initially converted to gray, then colors are inverted and finally a threshold is found.
Everything below this threshold is deleted (usually the background), while the rest of the image is kept intact. 
After that, it is found how many stars are in the image by calculating the number of objects.

### Number elaboration
When there are numbers on the image, the process is different because I have to identify and perform an operation (addition or subtraction) on the operands.
For this reason I decided to split the images into three parts and process them accordingly using a neural network.
The image elaboration starts in the same way as the previous step, by removing the background and cleaning the image by any possible disturb.
After that, I look for three different regions of pixels (ideally it should be two numbers and one operator).
Occasionally the operator is connected to the same region of one of the numbers, if it happens I have decided to "manually" split the biggest area to 5px starting from the rightmost limit.
This ensures to have a "usable" operator without cutting too much the operand.
Once recognized all the parts, the operation is performed and the resulting value returned.

## The Neural Network Part
I created two different neural networks to classify the two different type of input (stars and numbers).
As I wanted the program to be able to separate at the same time both the stars from the numbers and elaborate those in order to get the operation result, there are two neural networks:

1) One classify if the image belongs to the stars or the numbers class

2) Another recognize the numbers and the operators

All the NNs expect to work on 136x47 8-bit color png images, and they have the same structure, with two convolutional layers and three linear layers. 
The only difference between them is the number of linear units for the classification of the stars and the numbers/operator.
The details of the NN are in the neural_net.py file.

## Training offline
To train the neural network offline, it must be used the cli with the "train" option and: 
- dataset: specify the dataset on which train the neural network
- o: specify the output file

## Test offline
To test the neural network offline, it must be used the cli with the "classify" option and:
- b_dataset: specify the binary (numbers vs stars) dataset folder
- n_dataset: specify the number dataset folder
- b_nn: specify the binary trained neural network
- n_nn: specify the numbers trained neural network
- file: specify the file to classify

## WebService
I have also implemented a webservice with only one endpoint that allows the safe usage of the neural networks.
It can be found in the `restapi` module.
After lunching the webservice you can access to it with:
```
/classify
```
It accepts a JSON with the image encoded with a field called "base64_img" in witch there is the image encoded with BASE64
It will return the number of stars or the result of the operation. 


## Conclusions
I just put down some notes.

* The documentation is still a WIP, I will update this as soon as I can

* I will put some GH actions just to let everybody use the program with ease

* I will create soon a docker image as well

* The dataset I used is in the "dataset" folder

* The accuracy on the test set is around 1. It works pretty well :)
