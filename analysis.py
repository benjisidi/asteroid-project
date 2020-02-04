import matplotlib.pyplot as plt
import numpy as np
from astropy import stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture, DAOStarFinder

from helpers import read_fits


def crop_middle(array, length):
    """
    Crop a square of side 'length' out of an array
    If length is 0, does nothing
    """
    if length == 0:
        return array
    shape = array.shape
    row_mid = np.ceil((shape[0]-1)/2.)
    col_mid = np.ceil((shape[1]-1)/2.)
    crop_rows = np.indices((length, length))[0] + int(row_mid/2)
    crop_cols = np.indices((length, length))[1] + int(col_mid/2)
    crop = array[crop_rows, crop_cols]
    return crop


def calc_gain(flats, crop_length=0):
    """
    Find camera gain
    :param flats (str[][]): Pairs of filenames of debiased flat frames (for a range of intensities)
    :param crop_length (int): Use only a (centered) square of this side length
    cut out of the frames
    """
    sigs = []
    vars_ = []
    for pair in flats:
        pair_0 = read_fits(pair[0], return_array=True)
        pair_1 = read_fits(pair[1], return_array=True)
        mean_0 = np.mean(crop_middle(pair_0, crop_length))
        mean_1 = np.mean(crop_middle(pair_1, crop_length))
        mean_ratio = mean_0/mean_1
        pair_1_corrected = pair_1 * mean_ratio
        flat_corrected = pair_0 - pair_1_corrected
        std = np.std(crop_middle(flat_corrected, crop_length))
        var = (std**2)/2.
        sigs.append(mean_0)
        vars_.append(var)
    gain = np.polyfit(vars_, sigs, 1)[0]
    return gain


def find_sources(file_, fwhm):
    """
    Uses DAOStarFinder to extract source positions from fits file
    :param file_ (str): name of target .fits file
    :param fwhm (float): The full width half maximum of the gaussian kernel in pixels
    For more config see
    https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html
    """
    # Read in fits file as numpy array
    data = read_fits(file_, return_array=True)
    # Calculate background level
    mean, median, std = stats.sigma_clipped_stats(data)
    print(('mean', 'median', 'std'))
    print((mean, median, std))
    # Set up DAO Finder and run on bg subtracted image, printing results
    # sharplo=.2, sharphi=1., roundlo=-.3, roundhi=.3,
    daofind = DAOStarFinder(exclude_border=True, fwhm=fwhm, threshold=std)
    sources = daofind.find_stars(data - median)  # daofind(data-median) #
    print('Sources:')
    print(sources)
    # Save positions of detected sources to csv file
    positions = (sources['xcentroid'], sources['ycentroid'])
    print_positions = zip(
        *[sources[x] for x in [
            'id',
            'xcentroid',
            'ycentroid',
            'sharpness',
            'roundness1',
            'roundness2',
            'npix',
            'sky',
            'peak',
            'flux',
            'mag'
        ]])
    header = 'id,xcentroid,ycentroid,sharpness,roundness1,roundness2,npix,sky,peak,flux,mag'
    np.savetxt(file_[:-5]+'_positions.csv',
               print_positions, fmt='%.5e', header=header)
    # Show image with detected sources circled in blue
    apertures = CircularAperture(positions, r=4.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    plt.draw()
    # Scatter plot sharpness vs magnitude
    plt.figure(2)
    sharp, round_, mags = (sources['sharpness'],
                           sources['roundness1'], sources['mag'])
    plt.scatter(mags, sharp)
    plt.title('Sharpness vs Magnitude')
    plt.xlabel('Mag')
    plt.ylabel('Sharp')
    # Scatter plot roundness vs magnitude
    plt.figure(3)
    plt.scatter(mags, round_)
    plt.title('Roundness vs Magnitude')
    plt.xlabel('Mag')
    plt.ylabel('Roundness1')
    plt.show()
