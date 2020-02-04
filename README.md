# In this package

This package contains 5 python files.

- `helpers.py`: Common functions used in other files.
- `preprocessing.py`: Functions pertaining to converting and debaiasing images
- `analysis.py`: Functions for calculating sensor gain, and locating sources in images.
- `hough_transform.py`: Functions for finding straight lines using (unsurprisingly...) a hough transform. Also contains an "interactive" mode where the user can eyeball a sensible image threshold.
- `fitting.py`: Hefty function for fitting a tophat to an image region, and displaying the output.

# Dependencies & Installation

All dependencies are listed in `requirements.txt` and can be installed using `pip install -r requirements.txt`.

# Usage

This is incomplete work. It is intended that the user import whatever functions they see as useful from the various modules provided. The most incomplete section is the `fit` function in `fitting.py`.

It functions as follows:

1. Collapse the provided PSF and image along the given axis
2. Sum the given background level along the hight of the perpendicular axis using

`sigma_total = sqrt(height * sigma^2) = sqrt(height) * sigma`

3. Create data representing a tophat and convolve it with the PSF
4. Fit the left and right halves of the convolved tophat to the left and right halves of the image data.
5. Calculate the residuals for each point, and mask any areas whose residual is more than `2*sigma_total`
6. Re-fit the curve, ignoring the masked areas (their uncertainty is set to `np.inf`)
7. Print the results to the console and plot them on screen.

It currently suffers from three main shortfalls:

1. The fit is prone to falling into local minima, and is therefore dependent on a reasonably good initial guess.
2. I am not sure that `bg_sigma` is being used correctly: it appears far too low, masking almost everything
3. Since each half is fitted separately, and the asteroid trail is not very thick, your crop in the y-direction must be very well-centered to produce anything reasonable
