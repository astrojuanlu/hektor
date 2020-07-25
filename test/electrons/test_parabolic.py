import numpy as np
import pytest

from hektor.electrons.parabolic import getbetar


@pytest.mark.parametrize(
    "h, r, Jr, ptheta, expected_betar",
    [
        [
            np.array([3, 3.5]),
            np.array([0.52, 0.55]),
            np.array([1.2, 1.1]),
            np.array([0.1, 0.11]),
            np.array([0.058975519032537, 0.049668031380904]),
        ]
    ],
)
def test_getbetar(h, r, Jr, ptheta, expected_betar):

    betar = getbetar(h, r, Jr, ptheta)
    np.testing.assert_almost_equal(betar, expected_betar)
