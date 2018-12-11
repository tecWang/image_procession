import cv2

img = cv2.imread("../lung_gray.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# box filter
res = cv2.boxFilter(img, -1, (3,3))
cv2.imshow("boxfilter", res)
cv2.imwrite("./boxfilter_image.png", res)

# gussian filter
gussian_res = cv2.GaussianBlur(img,(3, 3),1.5)
cv2.imwrite("./gussian_res_image.png", gussian_res)


# median filter non-linear kernel
# median filter eliminates noise at the expense of loss of picture clarity
salt_img = cv2.imread("./salt_noisy.png")
med_res = cv2.medianBlur(salt_img, 3)
cv2.imwrite("./med_res_image.png", med_res)



cv2.waitKey(10000)
cv2.destroyAllWindows()