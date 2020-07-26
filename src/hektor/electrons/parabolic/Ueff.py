import numpy as np


def Ueff(h, phiz, Jr, ptheta):
    """
    Effective potential for the electron axial motion in the parabolic case.

    Parameters
    ----------

    h : numpy.ndarray
        effective plume radius at the axial position of the point(s)
    phiz : numpy.ndarray
        potential at the axis at those point(s)
    Jr, ptheta :
        momenta at those point(s)

    Returns
    -------

    phi : numpy.ndarray
        potential at the requested points
    """
    return -phiz + np.sqrt(2) * (Jr / np.pi + np.abs(ptheta)) / h
