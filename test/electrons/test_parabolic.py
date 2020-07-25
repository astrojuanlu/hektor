import numpy as np

from hektor.electrons.parabolic import getbetar


def test_getbetar():
    h = np.array([3, 3.5])
    r = np.array([0.52, 0.55])
    Jr = np.array([1.2, 1.1])
    ptheta = np.array([0.1, 0.11])

    expected_betar = np.array([0.058975519032537, 0.049668031380904])

    betar = getbetar(h, r, Jr, ptheta)
    np.testing.assert_almost_equal(betar, expected_betar)
