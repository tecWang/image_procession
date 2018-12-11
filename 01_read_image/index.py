import cv2 
import numpy as np

# read img
img = cv2.imread("../lung.png")

print("img.shape", img.shape)
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey(3000)
cv2.destroyAllWindows() 

# copy img
img2 = img.copy()
cv2.imwrite("./lung-copy.png", img2)