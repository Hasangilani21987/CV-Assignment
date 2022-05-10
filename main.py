# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read Image:-
# ReadImg = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)

# Display Image:-
# ReadImg = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("image", ReadImg)

# Save Image to disk:-
# ReadImg = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
# status = cv2.imwrite("New-photo.jpg", ReadImg)
# NewReadImg = cv2.imread("New-photo.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("New Image", NewReadImg)

# Resize Image:-
# ReadImg = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
# resized_Image = cv2.resize(ReadImg, (250, 250),
#                            interpolation=cv2.INTER_NEAREST)
# cv2.imshow("Resized Image", resized_Image)

# Get Image Shape:-
# img = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
#
# print("Height of Image is =====> ", img.shape[0])
# print("Width of Image is =====> ", img.shape[1])


# Put Text On Image:-

# img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
# text_Image = cv2.putText(img, "OpenCV", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 250, 0), 2)
#
# cv2.imshow("Text Image", text_Image)

# Draw a Line on an Image:-

# img = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
# line_Image = cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 2)
#
# cv2.imshow("Line Image", line_Image)

# Draw a Circle on an Image:-

# img = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
# circle_Image = cv2.circle(img, (500, 300), 40, (0, 250, 0), 3)
#
# cv2.imshow("Circle Image", circle_Image)

# Draw an Reactangle on an Image:-

# img = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
# rectangle_Image = cv2.rectangle(img, (110, 220), (390, 300), (0, 255, 0), 2)
#
# cv2.imshow("Circle Image", rectangle_Image)

# Draw an Square on an Image:-

# img = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
# square_Image = cv2.rectangle(img, (150, 150), (250, 250), (0, 255, 0), 2)
#
# cv2.imshow("square Image", square_Image)

# Convert Image to GrayScale:-

# img = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
# gray_Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("Gray Image", gray_Image)

# Convert GrayScaleImage to RGB:-

# img = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
# gray_Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# rgb_Image = cv2.cvtColor(gray_Image, cv2.COLOR_GRAY2RGB)
#
# cv2.imshow("RGB Image", rgb_Image)
# cv2.imshow("GRAY Image", gray_Image)


# Capture LiveStream From WebCam:-

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#     cv2.imshow('Input', frame)
#
#     c = cv2.waitKey(1)
#     if c == 27:
#         break
#
#
# cap.release()

# Read Video From Disk:-

# cap = cv2.VideoCapture('sample.mp4')
# if not cap.isOpened():
#     print("Error opening video file")
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         resized_Image = cv2.resize(frame, (350, 300),
#                                    interpolation=cv2.INTER_NEAREST)
#         cv2.imshow('Frame', resized_Image)
#     else:
#         break
#
# cap.release()

# Blur an Image:-

# ReadImg = cv2.imread('photo.jpg')
# blur_Img = cv2.blur(ReadImg, (10, 10))
# cv2.imshow("Original Image", ReadImg)
# cv2.imshow('blurred image', blur_Img)


# Detect Edges of Object in an Image:

# img = cv2.imread("python.png")
# edges = cv2.Canny(img, 100, 200)
#
# cv2.imshow("Original Image", img)
# cv2.imshow("Edge Detection ====> ", edges)


# Detect Contours in an Image:-

# img = cv2.imread("python.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# edge = cv2.Canny(gray, 100, 200)
#
# counters, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # print("Number of Contours found =====> " + str(len(counters)))
#
# # Draw all contours
# # -1 signifies drawing all contours
#
# cv2.drawContours(img, counters, -1, (0, 255, 0), 2)
#
# cv2.imshow('Contours', img)


# Crop an Image:-

# img = cv2.imread("python.png")
# cv2.imshow("Original Image", img)
#
# cropped_image = img[0:100, 0:200]
#
# cv2.imshow("cropped", cropped_image)

# Sharpen the Image:-

# img = cv2.imread("python.png")
# kernel = np.array([[-1, -1, -1],
#                    [-1, 9, -1],
#                    [-1, -1, -1]])
#
# sharpened = cv2.filter2D(img, -1, kernel)
# cv2.imshow('Original Image', img)
# cv2.imshow('Image Sharpening', sharpened)

# Apply an Identity Filter  on an Image:-

# img = cv2.imread("photo.jpg")
# kernel = np.array([[0, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 0]])
#
# identity = cv2.filter2D(img, -1, kernel)
# cv2.imshow('Identity Filter', identity)
# cv2.imshow('Original Image', img)

# Apply Gaussian Filter  on an Image:-

# img = cv2.imread('python.png')
# Gaussian_Filter = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
#
# cv2.imshow("Original Image", img)
# cv2.imshow('Gaussian_Filter Image', Gaussian_Filter)

# Apply Median Filter  on an Image:-

# ReadImg = cv2.imread('python.png')
# Median_Filter = cv2.medianBlur(ReadImg, 5)
# cv2.imshow("Original Image", ReadImg)
# cv2.imshow('Median Filter Image', Median_Filter)


# Apply Average Filter  on an Image:-

# img = cv2.imread('python.png')
# blur_Img = cv2.blur(img, (5, 5))
#
# cv2.imshow("Original Image", img)
# cv2.imshow('blurred image', blur_Img)

# Draw Histogram of an Image:-

# img = cv2.imread('photo.jpg')
# histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
#
# plt.plot(histogram)
# plt.show()

# Perform x-Negative of an Image:-

# img = cv2.imread('photo.jpg')
# img_neg = 1 - img
#
# cv2.imshow("Original Image", img)
# cv2.imshow("Negative Image", img_neg)


# Perform Thersholding of an Image:-

# img = cv2.imread('python.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
#
# cv2.imshow('Binary Threshold', thresh1)
# cv2.imshow('Binary Threshold Inverted', thresh2)
# cv2.imshow('Truncated Threshold', thresh3)
# cv2.imshow('Set to 0', thresh4)
# cv2.imshow('Set to 0 Inverted', thresh5)

# Apply Log Transformation on an Image:-

# img = cv2.imread('photo.jpg')
# c = 255 / np.log(1 + np.max(img))
# log_image = c * (np.log(img + 1))
#
# log_image = np.array(log_image, dtype = np.uint8)
#
# cv2.imshow("Original Image", img)
# cv2.imshow("Log Transformed Image", log_image)

# Apply Power Law Transformation on an Image:-

# img = cv2.imread('photo.jpg')
#
# gamma_corrected = np.array(255 * (img / 255) ** 2.2, dtype='uint8')
#
# cv2.imshow('gamma_transformed', gamma_corrected)


cv2.waitKey(0)
cv2.destroyAllWindows()
