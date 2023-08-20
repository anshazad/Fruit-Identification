# Fruit-Identification

Codes:

1. A_prototype_generation.py:
	This is just a simple program that takes the average of all the pixels of all the images of a particular class and stores the newly formed image in a separate folder. (folder name:prot_nodes)

2.A_prototype_generation execution.py:
	This program loads all the test images and checks their Euclidean distance from the prototype images formed earlier and returns the label corresponding to the minimum distance found then compare it with the actual label of the test image(which is its folder name) and then returns the accuracy by correctness/total_count/

3.A_hogImage.py
	In this variation, I extracted the hog feature image of all the test images and the prototype image and then compare them using Euclidean distance and again check the correctness by checking the returned label with the test label with the same formula as mentioned above

3.A_hogfeature.py
	In this variation, I am comparing the hog feature vector of the test image with the feature vector of all the prototype images and then finding the accuracy.

4.A_combined.py
	This is a combined code of all the above 3 codes for the A part. I did all the above computations and checking even if one model is giving the correct label then its correct. Hence we compare the accuracy.

**for using fewer train data I have taken images as an interval of 5 so that we could cover all the data by using fewer images.**

5.B_simple.py
	Just comparing the Euclidean distance of some train images from each class with all test images.

6.B_hogImage.py
	did the same loading and comparing as the above one but with the hog images.

7.B_feat2.py
	did the same loading and comparison as the above one but with a hog feature vector.

8.C_simple.py
	load all the data and made 2 arrays of test images and train images.
built the neural network and train it with the train images and validate it on test images.


** The accuracy and the correctness of the algorithm are given in each code


	 
	 
