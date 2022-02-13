import phconvert as phc
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def import_ptu_file(pfname, detector=None):
    """

    Import the required ptu file.

    Parameters
    ----------

    detector: (1 or 2)
        The detector the measurement was carried out on.

    Returns
    -------

    detectors: array
        The array containing the arrival photons.

     nanotimes: array
        Time the photons were recorded in units of nanotimes unit.
        
    timestamps: array
        Macrotimes of photon arrival.
        
    tags: OrderedDict
        File metadata.

    """

    # Find the file. Tell if file not found.

    filename = name + '.ptu'

    try:
        with open(filename):
            pass

    except IOError:
        raise Exception('ATTENTION: Data file not found, please check the filename.\n'
                        '           (current value "%s")' % filename)

    # Load the data in .ptu file along with its metadata.
    d = phc.loader.nsalex_pq(filename, pfname)[0]
    meta = phc.pqreader.load_ptu(filename)[3]
    m = meta.copy()
    tags = m.pop('tags')

    # Remove the memory overflow indicators from the imported arrays.
    valid = d['photon_data']['detectors'] != 127
    for field in ['detectors', 'timestamps', 'nanotimes']:
        d['photon_data'][field] = d['photon_data'][field][valid]

    # Remove any photons from detector 1, as we are using detector 2 for measurements.
    # Display the contents after this and make an array of the contents and counts.

    try:

        if detector == 1:

            for field in ('timestamps', 'nanotimes', 'detectors'):
                d['photon_data'][field] = d['photon_data'][field][d['photon_data']['detectors']!= 1]

        elif detector == 2:

            for field in ('timestamps', 'nanotimes', 'detectors'):
                d['photon_data'][field] = d['photon_data'][field][d['photon_data']['detectors'] != 0]

        else:
            raise ValueError

    except ValueError as e:

        raise Exception(
            'You need to specify whether you used detector 1 or detector 2.'
            ' This is done by setting detector_number to either 1 or 2.') from e

    # Assign the data more useful names.
    detectors = d['photon_data']['detectors']
    timestamps = d['photon_data']['timestamps']
    nanotimes = d['photon_data']['nanotimes']

    assert (np.diff(timestamps) >= 0).all()


    # Non-fully run images

    if tags['TTResult_StopReason']['value'] == 1:
        final_full_frame_no = np.max(np.where(detectors == 68))

        detectors = detectors[:final_full_frame_no]
        nanotimes = nanotimes[:final_full_frame_no]
        timestamps = timestamps[:final_full_frame_no]

    return detectors, nanotimes, timestamps, tags


@jit(nopython=True, parallel=True)
def convert(detectors, timestamps, nanotimes, x_pixels_value, y_pixels_value, detector=None):
    """

    Convert the outputs of import_ptu_file. Numba @jit(nopython=True, parallel=True) used to speed up the function.

    Parameters
    ----------

    detectors: array
        The array containing the arrival photons.

    timestamps: array
        Macrotimes of photon arrival.

    nanotimes: array
        Time the photons were recorded in units of nanotimes unit.

    x_pixels_value: int
        The number of pixels in the x direction.

    y_pixels_value: int
        The number of pixels in the y direction.

    detector: 1 or 2
        The detector the measurement was carried out on.

    Returns
    -------

    data: array
        The photons organised into their pixels binned in 3D array.

    """

    # To work out the time per pixel. Time at which 66 occurs less the time at which 65 occurs
    # divided by the number of pixels in the line.
    det_65 = np.where(detectors == 65)
    det_66 = np.where(detectors == 66)
    pixel_time = (timestamps[det_66] - timestamps[det_65]) / x_pixels_value

    # Create the data stack to hold the photons, the will be spatially sorted and arranged by their nanotime value.
    
    data = np.zeros((x_pixels_value, y_pixels_value, np.amax(nanotimes)), dtype=np.uint16)

    # Start with the variables as negative quantities, increment once the first 65 reached.
    # The line number in a frame.
    line_no = -1
    # The total line number.
    line_no_time = -1
    in_line = False

    # Set the value which corresponds to a detected photon.
    if detector == 1:

        photon_value = 0

    elif detector == 2:

        photon_value = 1

    # Spatially sort the photons into the data array.
    for detector_value, p_time, nanotime in zip(detectors, timestamps, nanotimes):

        # Start of a line.
        if detector_value == 65:

            # Increment the line tracking variables.
            line_no += 1
            line_no_time += 1

            # Value in timesteps corresponding to the start of a line.
            start_of_line_time = p_time

            # Set the inline checker to True.
            in_line = True

        # If in a line and the a photon is detected.
        elif in_line and (detector_value == photon_value):

            # Calculate the pixel in the line to which the photon belongs.
            pixel = int(np.floor((p_time - start_of_line_time) / (pixel_time[line_no_time])))

            # Ensure the photon is within the image and not noise.
            if pixel < x_pixels_value:
                # Add the photon the the correct pixel in the line and in the correct TCSPC bin.
                data[pixel][line_no][nanotime] += 1

        # End of line.
        elif detector_value == 66:

            # Set to no longer in line. Any photons now detected are noise and so not counted.
            in_line = False

        # End of frame.
        elif detector_value == 68:

            # Reset the line number, in the frame.
            line_no = -1

    return data


def bin_data(ndarray, ns):
    """

    Bins an ndarray in all axes based on the target shape, by summing.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Parameters
    ----------

    ndarray: array
        The array to be binned.

    ns: tuple
        The shape of the binned array.

    Returns
    -------
    
    ndarray: array
        The binned array.

    """

    return ndarray.reshape(ns[0], ndarray.shape[0] // ns[0], ns[1], ndarray.shape[1] // ns[1], ns[2], ndarray.shape[2] // ns[2]).sum((1, 3, 5))


def tcspc(data, tags, save=False):
    """

    Plot spatially integrated TCSPC curve.

    Parameters
    ----------

    data: array
        The data cube.

    tags: OrderedDict
        File metadata.

    save: bool, optional
        If 'False' the TCSPC curve will not be saved. PNG TCSPC will be saved if 'True'

    Returns
    -------

    None

    """

    tcspc_data = np.sum(data, axis=(0, 1))

    t = np.linspace(0, np.shape(data)[2], np.shape(data)[2]) * (tags['MeasDesc_Resolution']['value'] / 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(t, tcspc_data)

    ax.set_xlabel('Time (ns)')

    ax.set_ylabel('Intensity (counts)')

    ax.set_yscale('log')

    ax.minorticks_on()

    if save:
        plt.savefig(name + '_TCSPC_Curve.png', dpi=1000)
        np.savetxt(name + '_TCSPC_data.txt', list(zip(t, tcspc_data)))

    plt.show()

    return t, tcspc_data


def import_convert(fname, detector=None, TCSPC=False, save=False, pfname=True):
    """

    Carry out the initial processing.
    Imports the file with import_ptu_file, converts the data with convert.
    Uses tcspc to plot TCSPC curve.

    Parameters
    ----------

    fname: str
        The name of the file to import. Not including .ptu.

    detector: (1 or 2)
        The detector the measurement was carried out on.

    TCSPC: bool
        If True display the TCSPC curve.

    save: bool
        If True the TCSPC curve will be saved.

    Returns
    -------

    data_cube: array
        Spatially sorted arrival times from the imported ptu file.

    tags: OrderedDict
        File metadata.

    """
    global name
    name = fname

    detectors, nanotimes, timestamps, tags = import_ptu_file(pfname, detector=detector)

    x_pixels_value, y_pixels_value = tags['ImgHdr_PixX']['value'], tags['ImgHdr_PixY']['value']

    data_cube = convert(detectors, timestamps, nanotimes, x_pixels_value, y_pixels_value, detector=detector)

    data_cube = np.swapaxes((data_cube), 0, 1)

    if tags['ImgHdr_BiDirect']['value']: data_cube[1::2, ...] = data_cube[1::2, ...][:, ::-1, :]

    if TCSPC:
        tcspc(data_cube, tags, save)

    return data_cube, tags 


def plot_pl(data, tags, save=False, savetxt=False, plot=True):
    """

    Plot the PL intensity image.

    Parameters
    ----------

    data: array
        Spatially sorted arrival times from the imported ptu file.

    tags: OrderedDict
        File metadata.

    save: bool, optional
        If True save the image.
    
    savetxt: bool, optional
        If True save a txt file of the PL image.

    plot: bool, optional
        If true plot the PL image.

    Returns
    -------

    img: array
        The PL intensity image array.

    n: int
        Size of scale bar.

    len_in_pix: float
    	Number of pixels for the scale bar.
    """

    # The PL image. Flatten with sum along the TCSPC axis.
    img = data.sum(2)

    # The length of a pixel in 10^-6 m
    pixel_to_mum = tags['ImgHdr_PixResol']['value']

    # Best length of a scale bar. 13% of the length of the image.
    ideal_length_scale_bar = tags['ImgHdr_PixX']['value'] * pixel_to_mum * .133335

    # Work out how many pixels are required for the scalebar. If scale bar length is > 10 round to the nearest 5.
    if ideal_length_scale_bar > 10:

        n = (ideal_length_scale_bar - 10) / 5

        n = round(n)

        length = int(10 + 5 * n)

        len_in_pix = length / pixel_to_mum

    # Round to the nearest integer if between 1 and 10.
    elif (ideal_length_scale_bar <= 10) & (ideal_length_scale_bar >= 1):

        n = int(round(ideal_length_scale_bar))

        length = n

        len_in_pix = length / pixel_to_mum

    # Round to 1 decimal place if < 1.
    elif ideal_length_scale_bar < 1:

        n = round(ideal_length_scale_bar, 1)

        length = n

        len_in_pix = n / pixel_to_mum

    if plot:

        fig, ax = plt.subplots()

        pl_image = ax.imshow(img)

        ax.axis('off')

        plt.colorbar(mappable=pl_image, ax=ax)

        # Make the scalebar.
        scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + r' $\mathsf{\mu}$m', 3, pad=1,
                                borderpad=1, sep=2, frameon=False, size_vertical=1.5,
                                color='white', label_top=False)
        ax.add_artist(scalebar)

        # Save the PL image if required.
        if save:
            plt.savefig(name + '_PL.svg', dpi=300)

        # Save the PL image in txt file if required.
        if savetxt:
            np.savetxt(name + '_PL.txt', img)

        plt.show()

    return img, n, len_in_pix


def plot_fast_flim(data, tags, save=False, savetxt=False, plot=True):
    """

    Plot a fast flim image like the picoquant software.

    Parameters
    ----------
    
    data: array
        Spatially sorted arrival times from the imported ptu file.
    

    tags: OrderedDict
        File metadata.

    save: bool, optional
        If True save the image.
    
    savetxt: bool, optional
        If True save a txt file of the fast flim image.

    plot: bool, optional
        If true plot the fast flim image.

    Returns
    -------

    fast_flim: array
        The fast_flim data.

    """

    shape = data.shape

    argmax = data.sum((0, 1)).argmax()
    pl_img = data[..., argmax:].sum(2)
    pl_img[pl_img == 0] = 1
    idx = np.linspace(0, shape[-1]-argmax, shape[-1]-argmax, endpoint=False)
    t = idx * tags['MeasDesc_Resolution']['value']/1e-9
    fast_flim = (t[None, None, :]*data[..., argmax:]).sum(2)/(data[..., argmax:].sum(2))

    if plot:
        fig, ax = plt.subplots()

        ff_image = ax.imshow(fast_flim)

        ax.axis('off')

        clims = np.percentile(fast_flim, (1, 99.75))

        ff_image.set_clim(clims)

        plt.colorbar(mappable=ff_image, ax=ax)

        # The length of a pixel in 10^-6 m
        pixel_to_mum = tags['ImgHdr_PixResol']['value']

        # Best length of a scale bar. 13% of the length of the image.
        ideal_length_scale_bar = tags['ImgHdr_PixX']['value'] * pixel_to_mum * .133335

        # Work out how many pixels are required for the scalebar. If scale bar length is > 10 round to the nearest 5.
        if ideal_length_scale_bar > 10:

            n = (ideal_length_scale_bar - 10) / 5

            n = round(n)

            length = int(10 + 5 * n)

            len_in_pix = length / pixel_to_mum

        # Round to the nearest integer if between 1 and 10.
        elif (ideal_length_scale_bar <= 10) & (ideal_length_scale_bar >= 1):

            n = int(round(ideal_length_scale_bar))

            length = n

            len_in_pix = length / pixel_to_mum

        # Round to 1 decimal place if < 1.
        elif ideal_length_scale_bar < 1:

            n = round(ideal_length_scale_bar, 1)

            length = n

            len_in_pix = n / pixel_to_mum

        # Make the scalebar.
        scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + r' $\mathsf{\mu}$m', 3, pad=1,
                                borderpad=1, sep=2, frameon=False, size_vertical=1.5,
                                color='white', label_top=False)
        ax.add_artist(scalebar)

        # Save the fast film image if required.
        if save:
            plt.savefig(name + '_ff.svg', dpi=300)

        # Save the fast flim image in txt file if required.
        if savetxt:
            np.savetxt(name + '_ff.txt', fast_flim)

        plt.show()

    return fast_flim


def _1e(data, tags, time_binning):
    """

    Calculate the 1e times for each pixel in the image. Use numba to speed up the process.

    Parameters
    ----------

    data: array
        The data to caluclate the 1/e times for.

    tags: OrderedDict
        File metadata.
    
    time_binning: 
        The amount of binning to use in the time domain.

    Returns
    -------

    _1e: array
        The 1/e times.

    """

    # reshape to 2D array, easier to work with
    data_re = data.reshape(data.shape[0]*data.shape[1], -1)

    # get the index at which the maximum counts occur for each pixel
    argmax = data_re.argmax(1)

    # find where the data counts is not less than equal to 1/e counts
    a, b = ((data_re <= (data_re.max(1)*(1/np.e))[:, None]) == False).nonzero()

    # number of times counts <= 1/e counts
    counts = np.unique(a, return_counts=True)[1]

    # for next step. cumsum will give position of the last count which is > 1/e counts this position is in b where.
    pos = np.cumsum(counts) - 1

    # index difference of start time at each pixel, i.e. the max of tcspc curve, and the last counts > 1/e. times the time factor and rearranged.
    _1e = ((b[pos]-argmax)*((tags['MeasDesc_Resolution']['value'] * time_binning)/1e-9)).reshape(data.shape[0], data.shape[1])

    return _1e


def plot_1e_time(data, tags, time_binning=1, shape_binning=1, save=False, savetxt=False, plot=True):
    """

    Calculate the 1/e times using _1e and plot the resulting 1/e times.

    Parameters
    ----------

    data: array
        Spatially sorted arrival times from the imported ptu file.

    tags: OrderedDict
        File metadata.

    time_binning: int
        The amount of binning to use in the time domain.

    shape_binning: int
        The amount of binning to use in the spatial domain. i.e. with (256, 256)
        image shape_binning = 2 makes (128, 128).

    save: bool, optional
        If True save the image.
    
    savetxt: bool, optional
        If True save a txt file of the 1/e image.

    plot: bool, optional
        If true plot the 1/e times.

    Returns
    -------

    _1e_times: array
        The 1/e times.

    """

    data_sh = np.shape(data)

    if (time_binning != 1) or (shape_binning != 1):

        if data_sh[2] % time_binning != 0:

            data_sh_n = np.shape(data[:, :, :-(data_sh[2] % time_binning)])

            data = bin_data(data[:, :, :-(data_sh[2] % time_binning)], (int(data_sh_n[0] / shape_binning),
                                                                               int(data_sh_n[1] / shape_binning),
                                                                               int(data_sh_n[2] / time_binning)))

        elif data_sh[2] % time_binning == 0:

            data = bin_data(data, (int(data_sh[0] / shape_binning), int(data_sh[1] / shape_binning),
                                          int(data_sh[2] / time_binning)))
    
    _1e_times = _1e(data, tags, time_binning)

    if plot:

        fig, ax = plt.subplots()

        _1e_image = ax.imshow(_1e_times)

        ax.axis('off')

        plt.colorbar(mappable=_1e_image, ax=ax)

        # The length of a pixel in 10^-6 m
        pixel_to_mum = tags['ImgHdr_PixResol']['value'] * shape_binning

        # Best length of a scale bar. 13% of the length of the image.
        ideal_length_scale_bar = tags['ImgHdr_PixX']['value'] * pixel_to_mum * .133335 / shape_binning

        # Work out how many pixels are required for the scalebar. If scale bar length is > 10 round to the nearest 5.
        if ideal_length_scale_bar > 10:

            n = (ideal_length_scale_bar - 10) / 5

            n = round(n)

            length = int(10 + 5 * n)

            len_in_pix = length / pixel_to_mum

        # Round to the nearest integer if between 1 and 10.
        elif (ideal_length_scale_bar <= 10) & (ideal_length_scale_bar >= 1):

            n = int(round(ideal_length_scale_bar))

            length = n

            len_in_pix = length / pixel_to_mum

        # Round to 1 decimal place if < 1.
        elif ideal_length_scale_bar < 1:

            n = round(ideal_length_scale_bar, 1)

            length = n

            len_in_pix = n / pixel_to_mum

        # Make the scalebar.
        scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + r' $\mathsf{\mu}$m', 3, pad=1,
                                borderpad=1, sep=2, frameon=False, size_vertical=1.5 / shape_binning,
                                color='white', label_top=False)
        ax.add_artist(scalebar)

        # Save the 1e image if required.
        if save:
            plt.savefig(name + '_1e.svg', dpi=300)

        # Save the 1e image in txt file if required.
        if savetxt:
            np.savetxt(name + '_1e.txt', _1e_times)

        plt.show()

    return _1e_times
    
class interactive_tcspc:
    
    def __init__(self, t, data, int_ax, decay_ax, decay_fig, fig):
        
        self.data = data
        self.int_ax = int_ax
        self.decay_ax = decay_ax
        self.decay_fig = decay_fig
        self.int_ax.figure.canvas.mpl_connect('button_press_event', self.click)
        self.t = t
        self.y = None
        self.fig = fig
        
    def click(self, event):
        
        self.x0 = int(event.xdata)
        self.y0 = int(event.ydata)
        self.int_ax.patches = []
        self.y = self.data[self.y0, self.x0, :]
        self.decay_ax.set_data(self.t, self.y)
        self.int_ax.add_patch(Rectangle((self.x0 - .5, self.y0 - .5), 1, 1, fill = False, edgecolor = 'red'))
        self.fig.canvas.draw()
        
def plot_interactive_pl(data, tags, time_binning=1, shape_binning=1):
    """

    Plot PL image and a TCSPC curve which can be interactively adjusted.

    Parameters
    ----------


    data: array
        Spatially sorted arrival times from the imported ptu file.

    tags: OrderedDict
        File metadata.


    time_binning: int
        The amount of binning to use in the time domain.

    shape_binning: int
        The amount of binning to use in the spatial domain. i.e. with (256, 256)
        image shape_binning = 2 makes (128, 128).
        
    Returns
    -------

    None

    """    
   
    
    data_sh = np.shape(data)
    
    if (time_binning != 1) or (shape_binning != 1):

        if data_sh[2] % time_binning != 0:

            data_sh_n = np.shape(data[:, :, :-(data_sh[2] % time_binning)])

            data = bin_data(data[:, :, :-(data_sh[2] % time_binning)], (int(data_sh_n[0] / shape_binning),
                                                                               int(data_sh_n[1] / shape_binning),
                                                                               int(data_sh_n[2] / time_binning)))

        elif data_sh[2] % time_binning == 0:

            data = bin_data(data, (int(data_sh[0] / shape_binning), int(data_sh[1] / shape_binning),
                                          int(data_sh[2] / time_binning)))

    fig, ax = plt.subplots(1, 2, figsize=(9,4))

    img = ax[0].imshow(data.sum(2))
    ax[0].axis('off')
    fig.colorbar(img, ax=ax[0])

    # The length of a pixel in 10^-6 m
    pixel_to_mum = tags['ImgHdr_PixResol']['value'] * shape_binning

    # Best length of a scale bar. 13% of the length of the image.
    ideal_length_scale_bar = tags['ImgHdr_PixX']['value'] * (pixel_to_mum / shape_binning) * .133335

    # Work out how many pixels are required for the scalebar. If scale bar length is > 10 round to the nearest 5.
    if ideal_length_scale_bar > 10:

        n = (ideal_length_scale_bar - 10) / 5

        n = round(n)

        length = int(10 + 5 * n)

        len_in_pix = length / pixel_to_mum

    # Round to the nearest integer if between 1 and 10.
    elif (ideal_length_scale_bar <= 10) & (ideal_length_scale_bar >= 1):

        n = int(round(ideal_length_scale_bar))

        length = n

        len_in_pix = length / pixel_to_mum

    # Round to 1 decimal place if < 1.
    elif ideal_length_scale_bar < 1:

        n = round(ideal_length_scale_bar, 1)

        length = n

        len_in_pix = n / pixel_to_mum

    # Make the scalebar.
    scalebar = AnchoredSizeBar(ax[0].transData, len_in_pix, str(length) + r' $\mathsf{\mu}$m', 3, pad=1,
                            borderpad=1, sep=2, frameon=False, size_vertical=1.5,
                            color='white', label_top=False)
    ax[0].add_artist(scalebar)

    t = np.arange(data.shape[2]) * tags["MeasDesc_Resolution"]['value'] * time_binning * 1e9
    decay_img, = ax[1].plot(t, data.sum((0, 1)))
    ax[1].set_ylim(.1, data.sum((0, 1)).max() * 1.25)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Time (ns)')

    l = interactive_tcspc(t, data, ax[0], decay_img, ax[1], fig)
    plt.tight_layout()
    plt.show()
    
def fit_rate_eqn(data, tags, k1_array, k2_array, max_idx, shape, time_binning):

    from numba import prange
    from scipy.optimize import curve_fit
    
    def recom_model(t, k1, k2):
                    
        return ((n_0 * np.e**(-t*k1))/(1+(k2/k1)*n_0*(1-np.e**(-t*k1))))

    for i in prange(shape[0]):

        for j in prange(shape[1]):
            
            time = np.linspace(0, shape[2], num = shape[2], endpoint=False) * (tags['MeasDesc_Resolution']['value'] / 1e-9) * time_binning
            data[i, j] = data[i, j][data[i, j] != 0]
            t = time[data[i, j] != 0]
            offset = 250
            n_0 = data[i, j][max_idx[i, j] + offset]

            popt, pcov = curve_fit(recom_model, t[max_idx[i, j] + offset:], data[i, j][max_idx[i, j] + offset:],
                                 maxfev = 10000, xtol = 2.25e-16)

            k1_array[i, j] += popt[0]
            k2_array[i, j] += popt[1]

    return k1_array, k2_array


def rate_equation_plot(data, tags, time_binning=1, shape_binning=1, save=False, savetxt=False):

    data_sh = np.shape(data)

    binned_data = None

    if (time_binning != 1) or (shape_binning != 1):

        if data_sh[2] % time_binning != 0:

            data_sh_n = np.shape(data[:, :, :-(data_sh[2] % time_binning)])

            binned_data = bin_data(data[:, :, :-(data_sh[2] % time_binning)], (int(data_sh_n[0] / shape_binning),
                                                                               int(data_sh_n[1] / shape_binning),
                                                                               int(data_sh_n[2] / time_binning)))

        elif data_sh[2] % time_binning == 0:

            binned_data = bin_data(data, (int(data_sh[0] / shape_binning), int(data_sh[1] / shape_binning),
                                          int(data_sh[2] / time_binning)))

    if binned_data is not None:
        data = binned_data

    max_idx = np.argmax(data, axis=2)
    shape = np.shape(data)

    k_1_values = np.zeros((shape[0], shape[1]))
    k_2_values = np.zeros((shape[0], shape[1]))

    k1s, k2s = fit_rate_eqn(data, tags, k_1_values, k_2_values, max_idx, shape, time_binning)

    return k1s, k2s
