import cv2
import numpy as np

# load img
img = cv2.imread("../lung.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("../lung_gray.png", img)
print(img.shape)

# img:  source image
# n:    salt numbers
def salt(img, n):
    print("img.ndim", img.ndim)
    for i in range(n):
        # get x y position of salt noisy
        x = int(np.random.random() * img.shape[0])
        y = int(np.random.random() * img.shape[1])
        # confirm how to add noisy by the number of channels in the image
        if img.ndim == 2:
            img[x, y] = 255
        elif img.ndim == 3:
            img[x, y, 0] == 0
            img[x, y, 1] == 0
            img[x, y, 2] == 0
    return img

print(img.shape)
salt_img = salt(img, 1000)
cv2.imshow("salt", salt_img)
cv2.imwrite("./salt_noisy.png", salt_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()