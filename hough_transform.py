import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image
from scipy import ndimage
from skimage.transform import hough_line, hough_line_peaks

from helpers import read_fits


def threshold_image(array, cutoff, absolute=False):
    """
    Floor any data below a given threshold value.
    If absolute flag is true, set everything above the value to 1.
    """
    threshed = (array * [array >= cutoff])[0]
    if absolute:
        threshed[threshed >= cutoff] = 1
    return threshed


def hough_transform(file_name, threshold=None, crop={}, pad=0):
    """
    Use a hough transform on a file to find prominent straight lines
    Can threshold, crop and pad the image beforehand
    :param threshold (int): pixel value to threshold at
    :param crop (dict): coordinates of the crop (top left, bottom right): {"x1", "x2", "y1", "y2}
    :param pad (int): number of pixels of padding to add to each side
    """
    fits_image = read_fits(file_name, crop, pad, return_array=True)
    if threshold is not None:
        fits_image = threshold_image(fits_image, threshold)
    h, theta, d = hough_line(fits_image)
    # Find peaks in hough space
    hspace, angles, dists = hough_line_peaks(h, theta, d)
    return hspace, angles, dists


def threshold_image_interactive(image):
    """
    Displays an image and repeatedly asks the user for threshold values, displaying the result.
    """
    threshold = None
    while threshold is None:
        plt.imshow(image)
        print('Examine graph to determine background cutoff value.')
        time.sleep(1)
        plt.show()
        # Prompt for a threshold value and cut the image to that value
        while threshold is None:
            try:
                threshold = int(input("Enter threshold value: "))
            except ValueError:
                threshold = int(input("Please enter an integer: "))
        threshed_image = Image.fromarray(
            threshold_image(np.array(image), threshold))
        plt.imshow(threshed_image)
        plt.show()
        continue_ = input('Happy with cutoff? (Y/n) ').upper() != "N"
        if not continue_:
            threshold = None
    return threshed_image


def hough_transform_interactive(image_, crop_coords={}, convert_coords=False, pad=0):
    """
    Like threshold_image_interactive, but will display the result of a hough transform
    following thresholding and give the option to repeat the proceess
    """
    fits_image = read_fits(image_, crop_coords, convert_coords, pad)
    # Success counter used to allow the user to restart the process if
    # the hough transform returns too many lines and they wish to restart
    # with a new cutoff threshold
    success = 'n'
    while success.upper() != 'Y':
        thresh_image = threshold_image_interactive(fits_image)
        thresh_array = np.array(thresh_image)

        h, theta, d = hough_line(thresh_array)

        # Show the original image in greyscale
        fig, axes = plt.subplots(1, 3, figsize=(
            15, 6), subplot_kw={'adjustable': 'box'})
        ax = axes.ravel()
        gray_cmap = cm.get_cmap("gray")
        ax[0].imshow(thresh_image, cmap=gray_cmap)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        # Show the hough transform
        ax[1].imshow(np.log(1 + h),
                     extent=[np.rad2deg(theta[-1]),
                             np.rad2deg(theta[0]), d[-1], d[0]],
                     cmap=gray_cmap, aspect=8)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        # Show the original image with the detected lines overlaid in red
        ax[2].imshow(thresh_image, cmap=gray_cmap)
        [image_width, image_height] = thresh_array.shape
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image_height * np.cos(angle)) / np.sin(angle)
            ax[2].plot((0, image_height), (y0, y1), '-r')
        ax[2].set_xlim((0, image_height))
        ax[2].set_ylim((image_width, 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        plt.tight_layout()
        plt.show()

        # Ask user if they wish to start again
        success = input('Happy with result? (y/N) ')

    # Find peaks in hough space
    lines_ = hough_line_peaks(h, theta, d)
    # Get angles associated with peaks in degree format
    angles = [np.rad2deg(x) for x in lines_[1]]

    # For each detected line, rotate the original image such that that line is horizontal,
    # and display it along with the rotation angle
    fig, rotations = plt.subplots(1, len(angles))
    if len(angles) > 1:
        rt = rotations.ravel()
        for i in range(0, len(angles)):
            rt[i].set_title('Rotation angle: ' +
                            '{:f}'.format(angles[i] + 90) + ' degrees.')
            rt[i].imshow(ndimage.rotate(fits_image, angles[i] + 90))
    else:
        rotations.set_title('Rotation angle: ' +
                            '{:f}'.format(angles[0] + 90) + ' degrees.')
        rotations.imshow(ndimage.rotate(fits_image, angles[0] + 90))

    plt.show()
