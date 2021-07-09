"""Edge and corner detection using various algorithms. TODO: Estimate roof angles using Hough transform properly."""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from math import *
from matplotlib import patches
from scipy import ndimage

# Read image and apply a simple gaussian blur filter
image = cv.imread('images_2nd_set/image11.jpg', 0)
image_gaussian = cv.GaussianBlur(image, (3, 3), 0)
# Prewitt edge detection
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = prewitt_x.transpose()
image_prewitt_x = cv.filter2D(image, -1, prewitt_x)
image_prewitt_y = cv.filter2D(image, -1, prewitt_y)
image_prewitt = image_prewitt_x + image_prewitt_y
# Roberts edge detection
roberts_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
roberts_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
image_roberts_x = cv.filter2D(image_gaussian, -1, roberts_x)
image_roberts_y = cv.filter2D(image_gaussian, -1, roberts_y)
image_roberts = image_roberts_x + image_roberts_y
# Sobel edge detection
image_sobel_x = cv.Sobel(image_gaussian, cv.CV_8U, 1, 0, ksize=3)
image_sobel_y = cv.Sobel(image_gaussian, cv.CV_8U, 0, 1, ksize=3)
image_sobel = image_sobel_x + image_sobel_y
# Laplacian edge detection
laplacian_1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian_2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
image_laplacian_1 = cv.filter2D(image_gaussian, -1, laplacian_1)
image_laplacian_2 = cv.filter2D(image_gaussian, -1, laplacian_2)
# Laplacian of Gaussian (log) edge detection
log_mask = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]])
image_log = cv.filter2D(image_gaussian, -1, log_mask)
# Canny (probably the best results out of all if you adjust the thresholds) edge detection
image_canny = cv.Canny(image, 105, 205, apertureSize=3)

# Plot results
fig, axes = plt.subplots(3, 3)
titles = ['Original Image', 'Gaussian Blur', 'Prewitt', 'Roberts', 'Sobel', 'Laplacian v1', 'Laplacian v2',
          'Laplacian of Gaussian', 'Canny']
images = [image, image_gaussian, image_prewitt, image_roberts, image_sobel, image_laplacian_1, image_laplacian_2,
          image_log, image_canny]
indexes = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

for title, cur_image, cur_index in zip(titles, images, indexes):
    axes[cur_index[0], cur_index[1]].imshow(cur_image, cmap='gray'), axes[cur_index[0], cur_index[1]].set_title(title)
    axes[cur_index[0], cur_index[1]].set_xticks([]), axes[cur_index[0], cur_index[1]].set_yticks([])

plt.show()


# Estimate roof angles using Hough transform
hough_space, angles, distances = hough_line(image_canny[:, 200:325])
angle = []
for _, cur_angle, distances in zip(*hough_line_peaks(hough_space, angles, distances)):
    angle.append(cur_angle)

angles = [a * 180 / np.pi for a in angle]
print('Angles detected', angles)

# Billard image, estimate angle of the cue with respect to the horizontal axis. Apply Canny edge detector and proceed
# from there with Hough transform.
image = cv.imread('images_2nd_set/image31.png', 0)
kernel_size = 5
image = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
low_threshold = 50
high_threshold = 150
edges = cv.Canny(image, low_threshold, high_threshold)
# Hough transform
rho = 1  # Pixel distance in grid
theta = np.pi / 180  # Angle in radians
threshold = 15
min_line_length = 200  # Minimum number of pixels that constitute a line. To isolate the billard cue,
# use a relatively high number.
max_line_gap = 10  # Maximum gap in pixels between connectable line segments
line_image = np.copy(image) * 0  # Create an empty image to draw the detected lines on
lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
# Fill empty image with the detected lines
for line in lines:
    for x1, y1, x2, y2 in line:
        cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
print('Points of detected lines in the form of ((x1,y1), (x2,y2)):', lines)
lines_edges = cv.addWeighted(image, 1, line_image, 1, 0)

# Plot results (also display an image with the isolated billard cue)
fig, axes = plt.subplots(1, 4)
axes[0].imshow(image, cmap='gray'), axes[0].set_title('Original image')
axes[1].imshow(lines_edges, cmap='gray'), axes[1].set_title('Detected lines (Hough transform)')
axes[3].imshow(ndimage.rotate((lines_edges - image), -51), cmap='gray'), axes[3].set_title('Rotated image')
# Estimate angle between the horizontal axis and the billard cue.
horizontal_axis = (
    (0, int(image.shape[1] / 2) + 30), (image.shape[0], int(image.shape[1] / 2) + 30))  # Two points with same y
angle_1 = atan2(lines[0][0][1] - lines[0][0][3], lines[0][0][0] - lines[0][0][2])
angle_2 = atan2(horizontal_axis[0][1] - horizontal_axis[1][1], horizontal_axis[0][0] - horizontal_axis[1][0])
angle = np.degrees(abs(angle_1) - abs(angle_2))
# Add angle to image and plot it using an Arc object from matplotlib
cv.line(line_image, horizontal_axis[0], horizontal_axis[1], (255, 255, 0), 5)
lines_edges = cv.addWeighted(image, 0.8, line_image, 1, 0)
arc = patches.Arc(xy=(200, 280), width=80, height=80, angle=180, theta1=0, theta2=129, color='red')
axes[2].add_patch(arc)
axes[2].imshow(lines_edges, cmap='gray'), axes[2].set_title(
    'Estimated angle, angle=' + str(angle + 180))
plt.show()

