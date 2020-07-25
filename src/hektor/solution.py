import attr
import numpy as np


@attr.s(auto_attribs=True)
class Solution:
    """Solution structure.

    Parameters
    ----------
    npoints : int
        Number of points in the solution vector.
        First point must be origin. Last point must be infinity.
    h : numpy.ndarray
        Independent variable: plume characteristic radius at each test point.
        The first value must be 1; the final value must be infinity.
    r : numpy.ndarray
        Corresponding values of the radius for each test point.
    phi : numpy.ndarray
        Potential at each test point. Must be 0 at origin.
    ne00p : numpy.ndarray
        Density of the (vz > 0) electrons at the origin.

    """

    npoints: int
    h: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    ne00p: np.ndarray
