import os

import numpy as np
from astropy.io import fits

from helpers import read_fits
from pyraf import iraf


def x3f_to_fits(directory_):
    """Call sc_reduce on all unprocessed X3F files in the directory"""
    filelist = os.listdir(directory_)
    X3Fs = [x for x in filelist if x[-3:].upper() == 'X3F']
    for x in X3Fs:
        if x[:-3] + 'fits' not in filelist:
            os.system('sc_reduce -i ' + directory_ + x + ' -k')
    print(f'{len(X3Fs)} X3F files processed.')


def make_LUT(x):
    """Make Transmission to Density LUT from numpy array"""
    black = np.min(x)
    clear = np.max(x)
    LUT = np.empty(clear+1, dtype=np.float64)
    LUT[:] = np.NaN
    scale_factor = np.float64(clear/np.log10(clear-black))
    for j in range(black+1, clear+1):
        t = np.float64(j)
        d = np.log10(t - black)*scale_factor
        if d < 1.0:
            d = 1.0
        LUT[j] = (clear + 1 - int(d))
    for j in range(0, black+1):
        LUT[j] = black + 1
    index = np.arange(0, clear+1)
    data = zip(index, LUT)
    return LUT


def t2d(file_):
    """Read in a transmission fits file and output a copy converted to density"""
    if 'density' in file_:
        print(f'{file_} already converted.')
        return None
    array = read_fits(file_, return_array=True)
    LUT = make_LUT(array)
    density_array = np.copy(array)
    for x in np.nditer(density_array, op_flags=['readwrite']):
        x[...] = LUT[x]
    density_array = np.flipud(density_array)
    hdu = fits.PrimaryHDU(density_array)
    hdulist = fits.HDUList([hdu])
    try:
        hdulist.writeto(file_[:-5] + '_density.fits')
    except IOError:
        print(f'{file_} already converted.')


def gaia_to_imarith(file_):
    """Read in GAIA generated positions file and convert it to one usable by imarith()"""
    # Read the file outputted by GAIA
    gaia_file = open(file_, 'r')
    lines = gaia_file.readlines()
    # Remove extraneous lines, and split the remaining ones
    pos_lines = [x for x in lines if '#' not in x]
    columns = [x.split() for x in pos_lines]
    # Remove everything but the x, y coordinates
    positions = [x[1:3] for x in columns]
    # Write file with one set of x, y coordinates per line for use in IRAF
    outfile_name = file_[:-4] + '_imarith.txt'
    outfile = open(outfile_name, 'w')
    for line in positions:
        if line[0] != '\n':
            outfile.write(line[0] + ' ' + line[1] + '\n')
    # Return the name of the written file for later use
    return outfile_name


def make_master_bias(bias_frames):
    """Stack bias frames to make master bias"""
    iraf.imcombine(input='@'+bias_frames,
                   output='master_bias.fits', combine='average')


def debias_flats(flats, master_bias='master_bias.fits'):
    """
    Subract master bias from flats
    :param flats (str[][]): Pairs of flat frames taken at different exposure levels.
    Output for use in calc_gain (analysis.py)
    """
    for pair in flats:
        [flat_1, flat_2] = pair
        # Subtract bias from both images
        iraf.imarith(operand1=flat_1, operand2=master_bias,
                     op='-', result=flat_1[:-5]+'_debiased.fits')
        iraf.imarith(operand1=flat_2, operand2=master_bias,
                     op='-', result=flat_2[:-5]+'_debiased.fits')
