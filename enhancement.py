"""Examples of algorithmic techniques to increase brightness, enhance light/color and sharpen images"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image of dark forest
image = cv.imread('images_1st_set/pollen-500x430px-96dpi.jpg', 0)

# Calculate original image's histogram.
hist, bins = np.histogram(image.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Apply histogram equalization and display the images and their respective histograms.
# Use masking to handle missing values.
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf2 = np.ma.filled(cdf_m, 0).astype('uint8')
eq_image = cdf2[image]

# Plot images and histograms
fig, axes = plt.subplots(2, 2, figsize=(13, 3))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 1].plot(cdf_normalized, color='b')
axes[0, 1].hist(image.flatten(), 256, [0, 256], color='r')
axes[0, 1].set_xlim([0, 256])
axes[0, 1].legend(('cdf', 'histogram'), loc='center right')
axes[1, 0].imshow(eq_image, cmap='gray')
axes[1, 1].plot(cdf2, color='b')
axes[1, 1].hist(eq_image.flatten(), 256, [0, 256], color='r')
axes[1, 1].set_xlim([0, 256])
axes[1, 1].legend(('cdf', 'histogram'), loc='center right')
fig.suptitle('Histogram equalization example')
plt.show()

# Gamma adjustment
image = cv.imread('images_1st_set/nature_dark_forest.jpg')


def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values.
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')

    # Apply gamma correction using the lookup table
    return cv.LUT(image, table)


# Display gamma-adjusted images
fig, axes = plt.subplots(1, 4)
axes[0].imshow(image), axes[0].set_title('Original image'), axes[0].axis('off')
axes[1].imshow(adjust_gamma(image, gamma=0.65)), axes[1].set_title('Gamma = 0.65'), axes[1].axis('off')
axes[2].imshow(adjust_gamma(image, gamma=1.35)), axes[2].set_title('Gamma = 1.35'), axes[2].axis('off')
axes[3].imshow(adjust_gamma(image, gamma=1.90)), axes[3].set_title('Gamma = 1.90'), axes[3].axis('off')
plt.show()

# Sharpen an image using a high boost filter (unmask kernel)
image = cv.imread('images_1st_set/First-photo-of-the-moon-from-Chandrayaan-2_ISRO.jpg', 0)
sharpen_kernel = -np.ones((3, 3), np.float32) / 9
sharpen_kernel[1, 1] *= -10  # value of z in the middle of the kernel
sharpened_image = cv.filter2D(image, -1, sharpen_kernel, borderType=cv.BORDER_CONSTANT)
# Try another z value
sharpen_kernel = -np.ones((3, 3), np.float32) / 9
sharpen_kernel[1, 1] *= -14
sharpened_image_v2 = cv.filter2D(image, -1, sharpen_kernel, borderType=cv.BORDER_CONSTANT)
# Plot original and sharpened image
fig, axes = plt.subplots(1, 3)
axes[0].imshow(image, cmap='gray'), axes[0].set_title('Original image'), axes[0].axis('off')
axes[1].imshow(sharpened_image, cmap='gray'), axes[1].set_title('Sharpened image with high boost filter (z=10)'), axes[
    1].axis(
    'off')
axes[2].imshow(sharpened_image_v2, cmap='gray'), axes[2].set_title('Sharpened image with high boost filter (z=14)'), \
axes[2].axis(
    'off')
plt.show()
