import numpy as np
import scipy.signal as signal


def running_average(y: np.ndarray, kernelwidth: int) -> np.ndarray:
    """
    Compute the running average of a 1D array using a specified kernel width.
    Parameters
    ----------
    y : array_like
        Input array for which the running average is to be computed.
    kernelwidth : int
        Width of the averaging kernel. Must be a positive integer. If an even 
        value is provided, it will be incremented by 1 to ensure it is odd.
    Returns
    -------
    ndarray
        The smoothed array after applying the running average.
    Raises
    ------
    ValueError
        If `kernelwidth` is less than 1.
    Notes
    -----
    - The function pads the input array on both ends with the first and last 
      values of the array, respectively, to handle edge effects.
    - The kernel used for averaging is a uniform kernel with values summing to 1.
    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> running_average(y, kernelwidth=3)
    array([1.66666667, 2.        , 3.        , 4.        , 4.33333333])
    """

    if kernelwidth < 1:
        raise ValueError("running_average: kernelwidth must be greater than 0")
    if kernelwidth % 2 == 0:
        kernelwidth += 1
    kernel = np.ones(kernelwidth) / kernelwidth
    lpad = np.ones(kernelwidth) * y[0]
    rpad = np.ones(kernelwidth) * y[-1]
    padded = np.concatenate((lpad, y, rpad))
    y = np.convolve(padded, kernel, mode="same")[kernelwidth:-kernelwidth]
    return y


def eod_events(time, eod, threshold=0.0, running_avg=0):
    """Detect EOD events in the EOD signal.

    Parameters
    ----------
    time : np.ndarray
        The time axis of the EOD signal.
    eod : np.ndarray
        The EOD signal.
    threshold : float, optional
        The threshold for event detection, defaults to 0.0.
    running_average : int, optional
        Width of the running average that is to be applied before event detection,
        defaults to 0, no filtering.

    Returns
    -------
    np.ndarray
        The time points of the detected events.
    """
    eod -= np.mean(eod)
    if running_avg > 0:
        eod = running_average(eod, running_avg)
    events = time[(eod >= threshold) & (np.roll(eod,1) < threshold)]

    return events


def extract_am(y:np.ndarray, fs: float = 20000, lporder: int = 4, lpcutoff: float = 300) -> np.ndarray:
    """
    Extracts the amplitude modulation (AM) envelope of a signal.
    Parameters
    ----------
    y : np.ndarray
        Input signal as a 1D numpy array.
    fs : float, optional
        Sampling frequency of the input signal in Hz. Default is 20000.
    lporder : int, optional
        Order of the  Butterworth low-pass filter. Default is 4.
    lpcutoff : float, optional
        Cutoff frequency of the Butterworth low-pass filter in Hz. Default is 300.
    Returns
    -------
    np.ndarray
        The amplitude modulation envelope of the input signal.
    """

    y = y - np.mean(y)
    y[y < 0] = -1 * y[y < 0]
    sos = signal.butter(lporder, lpcutoff, 'lp', fs=fs, output='sos')
    am = signal.sosfiltfilt(sos, y)
    return am
