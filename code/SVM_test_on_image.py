import glob
import pandas as pd  # as means that we use pandas library short form  as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt # matplotlib is big library, we are just calling pyplot function 
                                    # for showing images
from skimage.feature import hog #We are calling only hog  

from sklearn.decomposition import PCA # Calling the PCA funtion from sklearn
from sklearn.svm import SVC # # Calling the SVM function from sklearn
from sklearn.externals import joblib # Calling the joblib function from sklearn, use for model saving 
                                     # and loading.
import sys                                     
# Loading the mode into same name
pca = joblib.load('pca.pkl')
classifier = joblib.load('svm.pkl')



from skimage.exposure import exposure #for displaying th hog image.
# img_path=csv_files_Testing[main_Testing['ClassId'][image_number]].split('GT')[0]+main_Testing['Filename'][image_number]
img_path=sys.argv[1]
#img_path="dataset/Testing/00014/00389_00001.ppm
print ('Reading Image from Path: ',img_path)
img = cv2.imread(img_path)
crop_image=img
img0=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img0 = cv2.medianBlur(img0,3)

crop_image0=img0
crop_image0=cv2.resize(crop_image0, (64, 64))

# Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
# need to calculate.
ret,crop_image0 = cv2.threshold(crop_image0,127,255,cv2.THRESH_BINARY)
descriptor,imagehog  = hog(crop_image0, orientations=8,pixels_per_cell=(4,4),visualize=True)


# descriptor,imagehog = hog(crop_image0, orientations=8, visualize=True)
descriptor_pca=pca.transform(descriptor.reshape(1,-1))
#
## Initilize the 3 axis so that we can plot side by side
#fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
#
##ploting crop image
#ax1.axis('off')
#ax1.imshow(cv2.cvtColor(crop_image,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
#ax1.set_title('Crop image')
#
## Rescale histogram for better display,Return image after stretching or shrinking its intensity levels
#hog_image_rescaled = exposure.rescale_intensity(imagehog, in_range=(0, 10))
##ploting Hog image
#ax2.axis('off')
#ax2.imshow(imagehog, cmap=plt.cm.gray)
#ax2.set_title('Histogram of Oriented Gradients')
##ploting Orignal image
#ax3.axis('off')
#ax3.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
#ax3.set_title('Orignal Image')
#plt.savefig('Result.png')
# class predition of image using SVM
Predicted_Class=classifier.predict(descriptor_pca)[0]
print ('Predicted Class: ',Predicted_Class)