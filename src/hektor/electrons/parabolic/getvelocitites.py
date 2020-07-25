import numpy as np


def getvelocities(h, r, phiz, E, Jr, ptheta):
    """ Returns |vz|, |vr|, |vtheta| at a point (h,r), given phiz, E, Jr and ptheta.

    Parameters
    ----------
    h : numpy.ndarray
        Effective plume radius at the axial position of the point(s)
    r : numpy.ndarray
        Radius of the point(s)
    phiz : numpy.ndarray
        Potential at the axis at those point(s)
    E, Jr, ptheta : numpy.ndarray
        Mechanical energy and momenta at those point(s)

    Returns
    -------
    absvz,absvr,absvtheta : numpy.ndarray
        Absolute value of the velocities
    """
    # Compute velocities
    # Axial velocity | vz |
    temp = np.sqrt(2) * (Jr / np.pi + np.abs(ptheta)) / h ** 2
    absvz = np.sqrt((E + phiz - temp) * 2)

    # Azimuthal velocity | vtheta | (definition of ptheta)
    absvtheta = np.abs(ptheta / r)
    absvtheta[np.isnan(absvtheta)] = 0  # vtheta = 0 always at the axis

    # Radial velocity | vr |
    absvr = np.sqrt((temp - (r ** 2) / h ** 4) * 2 - absvtheta ** 2)

    return absvz, absvr, absvtheta
