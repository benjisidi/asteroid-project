
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from helpers import read_fits, pluck


def make_tophat(length, step_up, step_down):
    """
    Makes tophat data of desired length.
    """
    step_length = step_down - step_up
    return [
        *[0 for x in range(int(step_up))],
        *[1 for x in range(int(step_length))],
        *[0 for x in range(int(length - step_down))]
    ]


def pretty_output(opt, cov, title=''):
    print(f"""
                        {title}
    -------------------------------------------------------
    -------------------------------------------------------

                    ---Optimal Values---
        B:  {opt[0]} +/- {cov[0][0]}
        A:  {opt[0]} +/- {cov[1][1]}
       x0:  {opt[0]} +/- {cov[2][2]}

                  ---Variance & Covariance---
{cov}
""")


def make_sigmas(length, sigma, sigma_extreme, outlier_indicies):
    output = []
    for i in range(length):
        if i in outlier_indicies:
            output.append(sigma_extreme)
        else:
            output.append(sigma)
    return output


def plot_result(x_data, image, fit, title, masked=[]):
    plt.figure()
    plt.plot(x_data, image, 'b-', label='data')
    plt.plot(x_data, fit, 'r-', label='fit')
    for px in masked:
        plt.axvspan(x_data[px] - .5, x_data[px] + .5,
                    color='red', alpha=.3, lw=0)
    plt.xlabel('coordinate')
    plt.ylabel('signal')
    plt.title(title)
    plt.legend()


def fit(image, psf, axis, bg_sigma=0, psf_scale_factor=1):
    """
    Marginalizes image and psf along given axis
    Creates a tophat the same length as the data
    Convolves psf with tophat
    Fits result to data

    :param image (array_like): 2D image data, cropped around the asteroid trail (can use read_fits for this)
    :param psf (array_like): 2D PSF data
    :param axis (int): Axis to fit over: 0 for x, 1 for y
    :param bg_sigma: The estimated background level, used for
    calculating residuals to the fit
    :param psf_scale_factor: If you are using a high-resolution PSF, it is
    likely that it needs to be "squashed" to fit the scale of the original image.
    This quantity should be actual/expected size in the direction you are interested.
    (eg, if your high-res psf is 70px wide, but rescaled should be 35, your psf_scale_factor is 2)
    """
    psf_data = np.sum(psf, axis=axis)
    image_data = np.sum(image, axis=axis)

    # Since we're summing down an axis, we need to also sum the bg sigma
    # This is sqrt(height * sigma**2) = sqrt(height)*sigma
    perp_height = image.shape[axis]
    background = np.sqrt(perp_height)*bg_sigma

    # Our "initial guess" for the positions of the step are
    # 10% of the way through the data, and 90% of the way through the data
    # since it assumed the input has been cropped to just include the trail with
    # little space around it.
    step_up = int(len(image_data) * 0.1)
    step_down = int(len(image_data) * 0.9)
    tophat_data = make_tophat(len(image_data), step_up, step_down)
    convolved_tophat = np.convolve(psf_data, tophat_data)
    normalized_tophat = np.divide(convolved_tophat, convolved_tophat.max())

    half_length = int(len(normalized_tophat) / 2)

    halves = {
        'left': {
            'half_tophat_data': normalized_tophat[0:half_length],
            'half_image_data': image_data[0:half_length],
            'zero_point': step_up,
            'fill_value': (0, 1)
        },
        'right': {
            'half_tophat_data': normalized_tophat[half_length:],
            'half_image_data': image_data[half_length:],
            'zero_point': step_down - half_length,
            'fill_value': (1, 0)
        }
    }

    for half in halves:
        half_tophat_data, half_image_data, zero_point, fill_value = pluck(
            halves[half], 'half_tophat_data', 'half_image_data', 'zero_point', 'fill_value')

        interpolation_x_data = np.linspace(
            - zero_point / psf_scale_factor,
            (len(half_tophat_data) - zero_point) / psf_scale_factor,
            len(half_tophat_data)
        )

        interpolated_step = interp1d(
            interpolation_x_data, half_tophat_data,
            kind="cubic",
            fill_value=fill_value,
            bounds_error=False
        )

        def tophat_function(x, B, A, x0):
            return B + A*interpolated_step(x - x0)

        # [initial_B, initial_A, initial_x0]
        guesses = [
            np.min(half_image_data), np.mean(half_image_data), zero_point
        ]
        image_x_data = range(len(half_image_data))
        opt, cov = curve_fit(tophat_function, image_x_data,
                             half_image_data, p0=guesses)

        pretty_output(opt, cov, f'Initial Fit - {half}')
        plt.ion()
        plot_result(
            x_data=image_x_data,
            image=half_image_data,
            fit=tophat_function(image_x_data, *opt),
            title=f'Initial Fit - {half}'
        )
        outliers = []
        residuals = []
        for i in range(0, len(image_x_data)):
            residual = np.abs(
                half_image_data[i] - tophat_function(image_x_data[i], *opt))
            residuals.append(residual)
            if residual > 2 * background:
                outliers.append(i)
        sigs = make_sigmas(len(image_x_data), background, np.inf, outliers)
        # Refit with outlier areas masked
        opt2, cov2 = curve_fit(tophat_function, image_x_data, half_image_data, p0=guesses,
                               sigma=sigs, absolute_sigma=True)
        pretty_output(opt2, cov2, f'After masking - {half}')
        plot_result(
            x_data=image_x_data,
            image=half_image_data,
            fit=tophat_function(image_x_data, *opt2),
            title=f'After masking - {half}',
            masked=outliers
        )
    plt.show()
    done = input('Press return to close.')
    return opt, cov, opt2, cov2


if __name__ == '__main__':
    # hough_transform_interactive(
    #     "psf/5523.fits", {"x1": 1024, "x2": 2017, "y1": 1562, "y2": 1609}, convert_coords=True)
    # hspace, angles, dists = hough_transform("psf/5523.fits", threshold=600,
    #                                         crop={"x1": 1024, "x2": 2017, "y1": 1562, "y2": 1609})
    # print(hspace, angles, dists)
    test_data = read_fits(
        'psf/5523.fits',
        crop_coords={
            "x1": 1024,
            "x2": 2017,
            "y1": 1562,
            "y2": 1609
        },
        return_array=True
    )
    psf_data = read_fits('./5523_moffat25_width35.psf.fits', return_array=True)
    fit(test_data, psf_data, 1, bg_sigma=32, psf_scale_factor=10)
