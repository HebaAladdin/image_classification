# Image_Classification_Module

This is a module for getting visual features and training a classifier given these visual features. The repo contains pre-trained visual models on Imagenet like (Densenet121, VGG16, Resnet50, and Inceptionv2).

## How to use:
- Make a dataset folder and copy the images you want to classify into it (Note: there should be a folder for each class and the images inside it)

- In train.py specify the images location for training and testing (you can split the images from a single folder into training and testing)

- In train_configs.py you can specifiy the visual model, training epochs, save model location, and batch size, and the dense net layers

- To start training run train.py


