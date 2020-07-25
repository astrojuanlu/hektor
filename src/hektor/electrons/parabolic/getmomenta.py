import numpy as np


def getmomenta(h, r, phiz, vz, vr, vtheta):
    """
    Return the mechanical energy E, and the momenta Jr, ptheta from velocity variables at points (h, r).

    Parameters
    ----------

    h : numpy.ndarray
        effective plume radius at the axial position of the point(s)
    r : numpy.ndarray
        radius of the point(s)
    phiz : numpy.ndarray
        potential at the axis at those point(s)
    vz, vr, vtheta : numpy.ndarray
        velocity components at those point(s)

    Returns
    -------

    E, Jr, ptheta : numpy.ndarray
        mechanical energy and momenta at those point(s)
    """

    # Energies and momenta
    # Definition of |ptheta|
    ptheta = np.abs(r * vtheta)

    # Energy
    E = (vz**2 + vr**2 + vtheta**2)/2 - phiz + r**2 / h**4

    # Jr
    Jr = (np.sqrt(1/2) * (E + phiz - vz**2/2) * h**2 - np.abs(ptheta)) * np.pi

    return (E, Jr, ptheta)
