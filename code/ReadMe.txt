Python Version: 3.6.5

Libraries Required:

> Pandas Version: 0.24.2
> OpenCv Version: 3.3.1
> Numpy Version: 1.16.3
> Matplotlib Version: 3.0.3
> skimage Version: 0.15.0
> sklearn Version: 0.20.3


For Running the code:

1. 1.HOG_with_svm_classifier.py Run the code using python 3.
	
		
		Run code# python 1.HOG_with_svm_classifier.py dataset/Training dataset/Testing
		Meaning> python code_file Training_directory Testing_directory
	
	The code for reading the dataset and training the classifier.
	
	- It will load the dataset from Dataset/Training,Dataset/Testing Folders.
	- It will Saving the classes Images in the folder "classes_images". if the folder doesn't exist then create it with the name "classes_images".
	- It will save the Trained PCA and SVM model in the same directory where code is running.



2. SVM_test_on_image.py Run the code using python 3.
	
		
		Run code# python SVM_test_on_image.py dataset/Testing/00014/00389_00001.ppm
		Meaning> python code_file Image_path
	
	The code for load the testing file of ppm and find its predicted class.
	
	- It will load the image file
	- Load the model from pca,svm
	- Predict the class of image




3.  2.detection_MSER_Contours_HOG_SVM_multiple_image_video.py Run the code using python3.

		
		Run code# python detection_MSER_Contours_HOG_SVM_multiple_image_video.py dataset/input
		Meaning> python code_file Input_Images_Folder
	
	The code for Input the Images and detect and classify the Street sign and save it as video.

	- It will load the SVM and PCA trained model from the same directory where code is running.
	- Load the image one by one and apply Svm and Pca.
	- Testing on Folder of Images and save it as video.


4.  2.detection_MSER_Contours_HOG_SVM_on_image.py Run the code using python3.

		
		Run code# python 2.detection_MSER_Contours_HOG_SVM_on_image.py dataset/input/image.033600.jpg
		Meaning> python code_file Input_Images
	
	The code for Input the Images and detect and classify the Street sign and save the output,test the model by any choice of your image and watch the result.

	- It will load the SVM and PCA trained model from the same directory where code is running.
	- Load the image apply Svm and Pca.




	



