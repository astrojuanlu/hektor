import numpy as np
from scipy.special import erf
from scipy.optimize import root_scalar


def ne00p_phiinfty_nobarriers(chi):
    # MATLAB fzero(@(phiinfty) eq1(phiinfty)-eq2(phiinfty),-5);
    # Ideal Python: root_scalar(eq1_minus_eq2, args=(chi,), x0=-5)
    sol = root_scalar(eq1_minus_eq2, args=(chi,), x0=-5)
    phiinfty = sol.root
    ne00p = 1 / eq2(phiinfty, chi)

    return ne00p, phiinfty


def eq1(phiinfty):
    return (
        1
        + erf(np.sqrt(-phiinfty))
        - np.sqrt((-2 * phiinfty) / np.pi) * np.exp(phiinfty)
    )


def eq2(phiinfty, chi):
    return np.sqrt(2 / np.pi) * (1 - phiinfty) * np.exp(phiinfty) / chi


def eq1_minus_eq2(phiinfty, chi):
    return eq1(phiinfty) - eq2(phiinfty, chi)
