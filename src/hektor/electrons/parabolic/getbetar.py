import numpy as np


def getbetar(h, r, Jr, ptheta):
    """Returns betar at a point (h,r), given Jr and ptheta.

    Parameters
    ----------
    h, r : numpy.ndarray
        Position of the point(s)
    Jr, ptheta : numpy.ndarray
        Momenta at those point(s)

    Returns
    -------
    betar : numpy.ndarray
        The value of the radial angle coordinate betar at those point(s)

    """

    betar = np.arccos(
        (Jr / np.pi + np.abs(ptheta) - np.sqrt(2) * r ** 2 / h ** 2)
        / np.sqrt(Jr / np.pi * (Jr / np.pi + 2 * np.abs(ptheta)))
    ) / (2 * np.pi)

    betar[np.isnan(betar)] = 0  # For Jr = ptheta = r = 0

    return betar
