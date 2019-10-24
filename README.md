# Face-Verification
This repository about the implementation of Deep Neural network model used for face verification. The dataset used for this task is IMDB-WIKI-face images Dataset. 

[Link](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

The dataset consists of 5,23,051 face images of 20,284 celebrities(4,60,273 imdb and rest from wiki)
The images with multiple faces or no faces in the dataset are not considered for the task.

# Challenges:
* The celebrities facial images are taken at multiple poses and their age vary across the images(multi-modal)

# Why Neural Networks?
 Face space has a manifold structure on pixel space(by manifold hypothesis), which can be adequately captured by linear transformations(Hu, Lu, and Tan 2014). So I am using neural networks to learn the function for facial verification.
 
 
 # Siamese Nets, Triplet Nets
These two kinds of networks used when labels are very few and for comparing positive labels and negative labels(ranking). only one image is available for 38,614 celebrities out of 50546 celebrities
 
 The idea behind a siamese network is that it takes two inputs which need to be compared each other, so we reduce each input into a latent vector representation and compare it using some standard arithmetic. In the case of the Triple Nets, we take three inputs one is the ground truth and compare it with one positive and one negative sample using some standard arithmetic.
 
 # Loss
 ## Pairwise Ranking Loss
 The objective is to learn representations with a small distance d between them for positive pairs and greater distance than some margin value m for negative pairs.
 
 ![](gif.gif)
 
 
