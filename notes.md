# Siamese network
The main idea behind a siamese network is that it takes two inputs which need to be 
compared to each other, so we reduce it to a denser and hopefully more "semantic" vector
representation and compare it using some standard arithmetic. Each input undergoes a  
dimensionality reduction transformation implemented as a neural network. Since we want 
the two images to be transformed in the same way, we train the two networks using shared weights.
The output of the dimensionality reduction is a pair of vectors, which are compared in some 
way to yield a metric that can be used to predict similarity between the inputs.

# One shot Classification requires that you have just one training example of 
# each class 


# Face verification
  * Exemplar SVMs
  * Siamese CNN
  * Deep ID

# Exempler SVMs
The idea is to train one linear SVM classifier, that is, a hyperplane separating our data, 
for each exemplar in the training set, so that we end up in each case with one positive instance 
and lots of negative ones.

First, we run our training set through a HOG descriptor. HOG descriptors are feature descriptors
based on gradient detection: The image is divided into cells, in which all pixels will "vote" for 
the preferred graident by means of an histogram(the weight of each pixel's vote is proportional to 
the graident magnitude). The resulting set of histograms is the descriptor, and it's been proven to 
be robust to many kinds of large-scale transformations and thus widely used in object and human detection 

The next step is to fit a linear SVM model for each positive example in the dataset. These SVMs will 
take as input only that positive example and the thousands of negative ones, and will try to 
find the hyperplane that maximizes the margin between them in the space using the validation set 
so that the best examples get pulled closer to our positivve--and the worst ones, further a apart
(without altering their order). From ther, given a new image, if we want to know whether it represents 
a given person, we can compute a compound score for it based on all of the person's exemplar SVMs, 
and decide on a threshold upon which to make our decision.

Image link": https://raw.githubusercontent.com/Joaoloula/sparse-face-verification/master/images/calibration.jpg
Link: https://joaoloula.github.io/face-verification.html

* Face Alginment: https://towardsdatascience.com/precise-face-alignment-with-opencv-dlib-e6c8acead262
* FAce Recognition: https://blog.netcetera.com/face-recognition-using-one-shot-learning-a7cf2b91e96c

# Extension
* Use MTCNN for detecting and aligning faces






