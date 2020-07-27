import numpy as np
import pytest

from hektor.electrons.parabolic import getbetar, getmomenta, getr, getvelocities, Ueff
from hektor.electrons.parabolic.semimaxwellian import ne00p_phiinfty_nobarriers


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


@pytest.mark.parametrize(
    "h, r, phiz, vz, vr, vtheta, expected_E, expected_Jr, expected_ptheta",
    [
        (
            np.float_(0.0),
            np.float_(0.0),
            np.float_(0.0),
            np.float_(0.0),
            np.float_(0.0),
            np.float_(0.0),
            np.nan,
            np.nan,
            0.0,
        ),  # To avoid ZeroDivisionError
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.8,
            np.array([0.6, 0.5]),
            0.0,
            0.0,
            0.0,
            0.0,
            np.array([0.8789062499999998, 0.6103515624999999]),
            np.array([1.2495608263570406, 0.8677505738590559]),
            np.array([0.0, 0.0]),
        ),
        (
            np.array([0.7, 0.6]),
            np.array([0.5, 0.4]),
            np.array([0.3, 0.2]),
            np.array([0.1, 0.05]),
            np.array([0.1, 0.05]),
            np.array([0.1, 0.05]),
            np.array([0.756232819658476, 1.0383179012345682]),
            np.array([0.9871939351512351, 0.9264747638411237]),
            np.array([0.05, 0.020000000000000004]),
        ),
    ],
)
def test_getmomenta(
    h, r, phiz, vz, vr, vtheta, expected_E, expected_Jr, expected_ptheta
):
    E, Jr, ptheta = getmomenta(h, r, phiz, vz, vr, vtheta)

    np.testing.assert_almost_equal(E, expected_E)
    np.testing.assert_almost_equal(Jr, expected_Jr)
    np.testing.assert_almost_equal(ptheta, expected_ptheta)


@pytest.mark.parametrize(
    "h, betar, Jr, ptheta, expected_r",
    [
        [
            np.array([3, 3.5]),
            np.array([0.058975519032537, 0.049668031380904]),
            np.array([1.2, 1.1]),
            np.array([0.1, 0.11]),
            np.array([0.52, 0.55]),
        ]
    ],
)
def test_getr(h, betar, Jr, ptheta, expected_r):
    r = getr(h, betar, Jr, ptheta)
    np.testing.assert_almost_equal(r, expected_r)


@pytest.mark.parametrize(
    "h, r, phiz, E, Jr, ptheta, exp_absvz, exp_absvr, exp_absvtheta",
    [
        [
            np.array([3, 3.5]),
            np.array([0.52, 0.55]),
            np.array([-0.4, -0.5]),
            np.array([2.0, 2.1]),
            np.array([1.2, 1.1]),
            np.array([0.1, 0.11]),
            np.array([1.746004254421979, 1.758907942005322]),
            np.array([0.328344867189611, 0.249421727347251]),
            np.array([0.192307692307692, 0.200000000000000]),
        ],
    ],
)
def test_getvelocities(h, r, phiz, E, Jr, ptheta, exp_absvz, exp_absvr, exp_absvtheta):
    absvz, absvr, absvtheta = getvelocities(h, r, phiz, E, Jr, ptheta)
    np.testing.assert_almost_equal(absvz, exp_absvz)
    np.testing.assert_almost_equal(absvr, exp_absvr)
    np.testing.assert_almost_equal(absvtheta, exp_absvtheta)


@pytest.mark.parametrize(
    "h, phiz, Jr, ptheta, expected_result",
    [
        (0.0, 0.0, 0.0, 0.0, np.nan),
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (0.7, np.array([0.5, 0.3]), 0.0, 0.0, np.array([-0.5, -0.3])),
        (
            np.array([0.6, 0.5]),
            np.array([0.4, 0.2]),
            np.array([0.2, 0.1]),
            np.array([0.05, 0.025]),
            np.array([-0.13209615044272438, -0.03925769026563464]),
        ),
    ],
)
def test_Ueff(h, phiz, Jr, ptheta, expected_result):
    result = Ueff(h, phiz, Jr, ptheta)

    np.testing.assert_almost_equal(result, expected_result)


@pytest.mark.parametrize(
    "chi, ne00p_exp, phiinfty_exp",
    [
        [0.1, 0.527038809116752, -2.760996633043687],
        [0.5, 0.717436009778448, -0.613999677882413],
    ],
)
def test_semimaxwellian_ne00p_phiinfty_nobarriers(chi, ne00p_exp, phiinfty_exp):
    ne00p, phiinfty = ne00p_phiinfty_nobarriers(chi)
    np.testing.assert_almost_equal(ne00p, ne00p_exp)
    np.testing.assert_almost_equal(phiinfty, phiinfty_exp)
