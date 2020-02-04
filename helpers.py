import numpy as np
from astropy.io import fits
from PIL import Image
from scipy import ndimage


def pluck(dict, *args):
    """Neatly destructure a dictionary."""
    return (dict[arg] for arg in args)


def convert_coordinates(coordinates, array):
    """When storing an image in an np array, (0,0) refers to the top-left pixel.
    When viewing an image in an application like GAIA, (0,0) refers to the bottom-left pixel.
    This converts between the two.
    """
    x1, x2, y1, y2 = pluck(coordinates, "x1", "x2", "y1", "y2")
    height = array.shape[0]
    y2_new = abs(y1 - height)
    y1_new = abs(y2 - height)
    return {"x1": x1, "y1": y1_new, "x2": x2, "y2": y2_new}


def crop_image(image, coordinates, convert_coords=False):
    """Crops an image to the given coordinates.
    Converts from 0 at bottom left to 0 at top left if required.
    Inputs can be PIL image or np array - returns the same type as given.
    """
    input_is_image = type(image) == Image.Image
    array = np.array(image) if input_is_image else image
    if convert_coords:
        coordinates = convert_coordinates(coordinates, array)
    x1, x2, y1, y2 = pluck(coordinates, "x1", "x2", "y1", "y2")
    cropped_array = array[y1:y2, x1:x2]
    if input_is_image:
        return Image.fromarray(cropped_array)
    else:
        return cropped_array


def read_fits(file_name, crop_coords={}, convert_coords=False, pad=0, return_array=False):
    """
    Convenient function to open a .fits image and return either a numpy array or PIL image object.
    Can also crop to desired coordinates and add (post-crop) padding if desired
    """
    hdulist = fits.open(file_name, ignore_missing_end=True)
    array = np.flipud(np.array(hdulist[0].data, dtype='uint16'))
    if crop_coords:
        array = crop_image(array, crop_coords, convert_coords)
    if pad > 0:
        array = np.pad(array, pad, 'constant')
    if return_array:
        return array
    else:
        return Image.fromarray(array)


def save_fits(array, filename_):
    array = np.flipud(array)
    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename_, overwrite=True)


def rotate_fits(fits_image, angle, output_file=None):
    if output == None:
        output = fits_image[:-5]+'rotated.fits'
    image = read_fits(fits_image)
    rotated = ndimage.rotate(image, angle)
    rotated_array = np.array(rotated)
    if output_file:
        save_fits(rotated_array, output_file)
    else:
        return rotated_array
