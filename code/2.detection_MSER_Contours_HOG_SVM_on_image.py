import glob
import pandas as pd  # as means that we use pandas library short form  as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt # matplotlib is big library, we are just calling pyplot function 
from skimage.feature import hog #We are calling only hog  
from sklearn.externals import joblib # Calling the joblib function from sklearn, use for model saving 
                                     # and loading.
import sys
np.warnings.filterwarnings('ignore')

# Loading the mode into same name
pca = joblib.load('pca.pkl')
classifier = joblib.load('svm.pkl')


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y
def image_fill(Binary_image):
    # Mask used to flood filling.
    im_th=Binary_image.astype('uint8').copy()
    h, w = im_th.shape[:2]
    im_floodfill = im_th.copy()
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 1);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    im_out[im_out==254]=0
    return im_out

def cnts_find(binary_image_blue,binary_image_red):
    cont_Saver=[]
    
    (_,cnts, _) = cv2.findContours(binary_image_blue.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#finding contours of conected component
    for d in cnts:
         if cv2.contourArea(d)>700:
                (x, y, w, h) = cv2.boundingRect(d)
                if ((w/h)<1.21 and (w/h)>0.59 and w>20):
                    cont_Saver.append([cv2.contourArea(d),x, y, w, h])
    
    (_,cnts, _) = cv2.findContours(binary_image_red.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#finding contours of conected component
    for d in cnts:
         if cv2.contourArea(d)>700:
                (x, y, w, h) = cv2.boundingRect(d)
                if ((w/h)<1.21 and (w/h)>0.59 and w>20):
                    cont_Saver.append([cv2.contourArea(d),x, y, w, h])
    return cont_Saver


#image_path='dataset/input/image.033640.jpg' # Tested image path
image_path=sys.argv[1]
print ('Reading Image from ',image_path)

img = cv2.imread(image_path)
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_rgb[:,:,0] = cv2.medianBlur(img_rgb[:,:,0],3) #applying median filter to remove noice
img_rgb[:,:,1] = cv2.medianBlur(img_rgb[:,:,1],3) #applying median filter to remove noice
img_rgb[:,:,2] = cv2.medianBlur(img_rgb[:,:,2],3) #applying median filter to remove noice

arr2=img_rgb.copy()
arr2 = cv2.normalize(arr2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

imgr=arr2[:,:,0]
imgg=arr2[:,:,1]
imgb=arr2[:,:,2]
# Calling the imgadujust function
imgr=imadjust(imgr,imgr.min(),imgr.max(),0,1) 
imgg=imadjust(imgg,imgg.min(),imgg.max(),0,1)
imgb=imadjust(imgb,imgb.min(),imgb.max(),0,1)
#4Normalize intensity of the red channel
Cr = np.maximum(0,np.divide(np.minimum((imgr-imgb),(imgr-imgg)),(imgr+imgg+imgb)))
Cr[np.isnan(Cr)]=0
#Normalize intensity of the blue channel
Cb = np.maximum(0,np.divide((imgb-imgr),(imgr+imgg+imgb)))
Cb[np.isnan(Cb)]=0

[rows,cols]=img[:,:,1].shape
#Red color, normalization then thresholding it as 1
sc=(cv2.normalize(Cr.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int')
mser = cv2.MSER_create(_min_area=100,_max_area=10000)
regions, _ = mser.detectRegions(sc.astype('uint8'))
BMred=np.zeros((rows,cols))
if len(regions)>0:
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            BMred[regions[i][j][1],regions[i][j][0]]=1
        

#Blue color, normalization then thresholding it as 1
sb=(cv2.normalize(Cb.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int')
mser = cv2.MSER_create(_min_area=100,_max_area=10000)
regions, _ = mser.detectRegions(sb.astype('uint8'))
BMblue=np.zeros((rows,cols))
if len(regions)>0:
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            BMblue[regions[i][j][1],regions[i][j][0]]=1
        
        
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# hsv color 

## Hsv range for red
s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
s[s<0.5]=0
s[s>0.65]=0
s[s>0]=1

v[v<0.2]=0
v[v>0.75]=0
v[v>0]=1
redmask=np.multiply(s,v)

## Hsv range for blue
s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
s[s<0.45]=0
s[s>0.80]=0
s[s>0]=1

v[v<0.35]=0
v[v>1]=0
v[v>0]=1
bluemask=np.multiply(s,v)

# Taking the common part that is in both.
BMred_mask=np.multiply(BMred,redmask)
BMblue_mask=np.multiply(BMblue,bluemask)

# filling the area connected
BMred_fill=image_fill(BMred_mask)
BMblue_fill=image_fill(BMblue_mask)


cont_Saver=cnts_find(BMblue_fill,BMred_fill)
print ("Total Contours Found: ",len(cont_Saver))
if len(cont_Saver)>0:
    cont_Saver=np.array(cont_Saver)

    cont_Saver=cont_Saver[cont_Saver[:,0].argsort()].astype(int)
    for conta in range(len(cont_Saver)):
        cont_area,x, y, w, h=cont_Saver[len(cont_Saver)-conta-1]

        #getting the boundry of rectangle around the contours.

        image_found=img[y:y+h,x:x+w]

        crop_image=image_found.copy()
        img0=cv2.cvtColor(image_found, cv2.COLOR_RGB2GRAY)
        img0 = cv2.medianBlur(img0,3)

        crop_image0=cv2.resize(img0, (64, 64))

        # Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
        # need to calculate.
        ret,crop_image0 = cv2.threshold(crop_image0,127,255,cv2.THRESH_BINARY)
        descriptor,imagehog  = hog(crop_image0, orientations=8,pixels_per_cell=(4,4),visualize=True)


        # descriptor,imagehog = hog(crop_image0, orientations=8, visualize=True)
        descriptor_pca=pca.transform(descriptor.reshape(1,-1))

        # class predition of image using SVM
        Predicted_Class=classifier.predict(descriptor_pca)[0]


        if Predicted_Class !=38:
            print ('Predicted Class: ',Predicted_Class)
            ground_truth_image=cv2.imread('classes_images/'+str(Predicted_Class)+'.png')

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)#drawing a green rectange around it.
            #Putting text on the upward of bounding box
            cv2.putText(img, 'Class: '+str(Predicted_Class), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 6)

            #loading the ground truth class respective to the predicted class
            # displaying the ground truth image
            #ground truth image resize and match according to the sign detected

            try:

                ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                #replaceing the sign image adjacent(left side) to detected sign
                img[y:y+ground_truth_image_resized.shape[0], x-w:x-w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized
            #if sign detected on left boundry then there will be an error because n0 place for image to place then this program run place the image one right side. 
            except:
                #ground truth image resize and match according to the sign detected
                ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                #replaceing the sign image adjacent(right side) to detected sign
                img[y:y+ground_truth_image_resized.shape[0], x+w:x+w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized

            print ('Saving Image as Final_Ouput.png')
            cv2.imwrite('Final_Ouput.png',img)
        else:
            print ('Saving Image as Final_Ouput.png')
            cv2.imwrite('Final_Ouput.png',img)
            


