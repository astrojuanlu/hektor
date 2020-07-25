import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from hektor.potential.parabolic.phi import phi


@pytest.mark.parametrize(
    "h,r,phiz,expected_result",
    [
        (np.float_(0.0), np.float_(0.0), 0.0, np.nan),  # To avoid ZeroDivisionError
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.1, 0.0, -0.01),
        (1.0, np.array([0.5, 0.2]), 0.0, np.array([-0.25, -0.04])),
        (
            np.array([1.0, 0.5]),
            np.array([0.5, 0.2]),
            np.array([0.0, 0.1]),
            np.array([-0.25, -0.54]),
        ),
    ],
)
def test_phi_sample_values(h, r, phiz, expected_result):
    result = phi(h, r, phiz)

    assert_array_almost_equal(result, expected_result)
