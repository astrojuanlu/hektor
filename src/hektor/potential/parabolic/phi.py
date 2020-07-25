import numpy as np


def phi(h, r, phiz):
    """
    Compute phi in the parabolic potential case.

    Parameters
    ----------
    h : numpy.ndarray
        effective plume radius at the axial position of the point(s)
    r : numpy.ndarray
        radius of the point(s)
    phiz : numpy.ndarray
        potential at the axis at those point(s)

    Returns
    -------
    phi : numpy.ndarray
        potential at the requested points
    """

    return -r**2 / h**4 + phiz
