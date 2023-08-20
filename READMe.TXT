Codes:

1. A_prototype_generation.py:
	This is just a simple program which takes the average of all the pixels of all the image of a particular class and store the newly formed image in a seperate folder.(folder name:prot_nodes)

2.A_prototype_generation execution.py:
	This program loads all the test images and check their euclidean distance from the prototype images formed eralier and return the label correspond to minimum distance found then compare it with the actuak label of the test image(which is it's folder name) and then returns the accuracy by correctness/total_count/

3.A_hogImage.py
	In this variation I extracted the hog feature image of all the test image and the prototype image and then compare it using euclidean diatance and again check the correctness by checking the returned label with the test label with the same formula as mentioned above

3.A_hogfeature.py
	In this variation I am comparing the hog feature vector of the test image with the feature vector of all the prototype images and then finding the accuracy.

4.A_combined.py
	This is combined code of all the above 3 codes for the A part. I did all the above computations and checking even if one model is giving the correct label then its correct. Hence we compare the accuracy.

**for using less train data I have taken images as an interval of 5 so that we could cover all the data with using less images.**

5.B_simple.py
	Just comparing the the eucledian diatance of some train images from each class with all test images.

6.B_hogImage.py
	did the same loading and comparing as above one but with the hog images.

7.B_feat2.py
	did the same loading and comparision as above one but with hog feature vector.

8.C_simple.py
	load all the data and made 2 arrays of test images and train images.
built the neural network and train it with the train images and validate on test images.


** the accuracy and the correctnes of the algorithm is given in each code


	 
	 