# Random String Recognition

## Objective
This section is focused on random string recognition using end-to-end learning of a custom CNN. This method wont be much useful in most of the practical applications (may be you could give try to break random letter captcha :wink: ) but definitely it is most simple step toward Optical Character Recognition and hence it will be a good starting excercise for a beginner. By doing this mini-project you learn following:
* Data generation
* Converting image data to tfrecord
* Verfying tfrecord visually
* Designing, training and testing a custom CNN network 

> Note: This is just a sample program to provide you template for random string recognition. You could definetily add more variation in dataset and use better CNN architecture to do much better.

## Overview
We will target to do following:
  1. Generate a simple dataset of random string image with following contraints:
      1. All images are of same size
      2. All images are single channel image with white background and black text
      3. Maximum number of character in random string is 16 (including whitespaces)
      4. Font used are one available from openCV (might be changed later to use any font)
  2. Convert dataset to tfrecord format
  3. Test the tfrecord file by reading it back and visulaizing the images and labels from dataset
  4. Train Custom CNN to recognize a random string with constraints specified in point 1
  5. Test the trained custom CNN model 

## Blog Posts
Link to the blog post: 
* Random string data generation:  https://medium.com/@vijendra1125/ocr-part-1-generate-dataset-69509fbce9c1
* Custom CNN training for random string recognition: https://medium.com/@vijendra1125/ocr-part-2-ocr-using-cnn-f43f0cee8016
