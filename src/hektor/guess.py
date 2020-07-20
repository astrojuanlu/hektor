import numpy as np

from hektor.solution import Solution


def generate_initial_guess(config):
    """Generate initial guess from config.

    """
    npoints = config["solution"]["npoints"]
    return Solution(
        npoints=npoints,
        h=np.append(np.linspace(1, 5, npoints - 1), np.inf),
        r=np.zeros(npoints),
        phi=np.linspace(0, -4, npoints),
        ne00p=0.51,
    )
