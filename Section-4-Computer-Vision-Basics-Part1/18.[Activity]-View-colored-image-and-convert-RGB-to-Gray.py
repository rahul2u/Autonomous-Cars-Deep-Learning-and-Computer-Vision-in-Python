import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Read Image,show and size

color_img = mpimg.imread('C:/Users/rahul/Autonomous-Cars-Deep-Learning-and-Computer-Vision-in-Python/images/road-1072823_640.jpg')
plt.imshow(color_img)
plt.show()
print(color_img.shape)

# show all image pixels
np.set_printoptions(threshold=sys.maxsize)
print(color_img)

# converting to color image to grayscale image
gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
print(gray_img)
plt.imshow(gray_img,cmap='gray')
plt.show()

# save color image to graw image
cv2.imwrite('C:/Users/rahul/Autonomous-Cars-Deep-Learning-and-Computer-Vision-in-Python/images/gray_image_road.jpg',gray_img)

