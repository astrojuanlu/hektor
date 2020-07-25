import numpy as np


def getr(h, betar, Jr, ptheta):
    """Returns r for a given h, betar, Jr and ptheta. It is the inverse function of getbetar.

    Parameters
    ----------
    h : numpy.ndarray
        Effective plume radius at the axial position of the point(s)
    betar : numpy.ndarray
        Radial angle coordinate at those point(s)
    Jr, ptheta : numpy.ndarray
        Momenta at those point(s)

    Returns
    -------
    r : numpy.ndarray
        Radius of the point(s)

    """
    r = np.sqrt(
        (h ** 2 / np.sqrt(2))
        * (
            Jr / np.pi
            + np.abs(ptheta)
            - np.cos(2 * np.pi * betar)
            * np.sqrt(Jr / np.pi * (Jr / np.pi + 2 * np.abs(ptheta)))
        )
    )

    return r
