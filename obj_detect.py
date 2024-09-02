import cv2
import numpy as np
image = cv2.imread(r'D:\Ramit\orinson_task\image(2).jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# Convert the image to grayscale

blurred = cv2.GaussianBlur(gray, (5, 5), 0)# Applying GaussianBlur to reduce noise

_, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)# binary thresholding of a binary image

contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Find contours in the binary image

cv2.drawContours(image, contours, -1, (0, 255, 0), 3)# Drawing contours on the original image

output=cv2.imshow('Object Detection', image)#display the result
cv2.waitKey(0)
# saving the image
cv2.imwrite('D:\Ramit\orinson_task\output_image(2).jpg', image)
