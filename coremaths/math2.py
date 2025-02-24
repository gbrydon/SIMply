"""
Module containing various useful mathematical functions.
"""

import math
import numpy as np
from typing import Callable, Tuple, Union, Optional

import coremaths.frame

FloatOrNP = Union[float, np.ndarray]


def quadSolve(a: FloatOrNP, b: FloatOrNP, c: FloatOrNP) -> Tuple[Optional[FloatOrNP], Optional[FloatOrNP]]:
    """ Solves a quadratic.

    :param a: x^2 coefficient
    :param b: x^1 coefficient
    :param c: x^0 coefficient
    :return: the roots of the quadratic.
    """
    if type(a) != np.ndarray:
        if a == 0:
            return None, None
    determinant = (b ** 2) - (4 * a * c)
    r1 = (-b + (determinant ** 0.5)) / (2 * a)
    r2 = (-b - (determinant ** 0.5)) / (2 * a)
    return r1, r2


def sphericalAreaIntegration(integrand: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
                             alims: Tuple[float, float], plims: Tuple[float, float], r=1, an=1000, pn=1000) -> float:
    """ Integrates an integrand over the surface of a sphere, with respect to area.

    :param integrand: the integrand (i.e. the function being integrated). This must take azimuth, polar angle and sphere
                    radius as arguments and return a value calculated from these inputs. It must be able to take azimuth
                    and polar angle as numpy arrays and provide its return value as a numpy array of the same shape.
    :param alims: the azimuth limits of the integration [radians]
    :param plims: the polar limits of the integration [radians]
    :param r: the radius of the sphere
    :param an: the number of discrete azimuth values used in the integration
    :param pn: the number of discrete polar values used in the integration
    :return: the result of the integration
    """
    a1, a2 = alims
    p1, p2 = plims
    da = (a2 - a1) / (an - 1)
    dp = (p2 - p1) / (pn - 1)
    a, p = np.meshgrid(np.linspace(a1, a2, an), np.linspace(p1, p2, pn))
    dArea = r * r * np.sin(p) * dp * da
    return float(np.sum(integrand(a, p, r) * dArea))


def sphericalSolidAngleIntegration(integrand: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
                             alims: Tuple[float, float], plims: Tuple[float, float], r=1, an=1000, pn=1000):
    """ Integrates an integrand over the surface of a sphere, with respect to solid angle.

    :param integrand: the integrand (i.e. the function being integrated). This must take azimuth, polar angle and sphere
                    radius as arguments and return a value calculated from these inputs. It must be able to take azimuth
                    and polar angle as numpy arrays and provide its return value as a numpy array of the same shape.
    :param alims: the azimuth limits of the integration [radians]
    :param plims: the polar limits of the integration [radians]
    :param r: the radius of the sphere
    :param an: the number of discrete azimuth values used in the integration
    :param pn: the number of discrete polar values used in the integration
    :return: the result of the integration
    """
    a1, a2 = alims
    p1, p2 = plims
    da = (a2 - a1) / (an - 1)
    dp = (p2 - p1) / (pn - 1)
    a, p = np.meshgrid(np.linspace(a1, a2, an), np.linspace(p1, p2, pn))
    dSolidAng = np.sin(p) * dp * da
    return float(np.sum(integrand(a, p, r) * dSolidAng))


def binNumpyArray2D(arr: np.ndarray, factor: int) -> np.ndarray:
    """Returns the result of segmenting a 2D array into 'factor'-by-'factor' shaped sub-arrays and calculating the total
    value within each sub-array.

    :param arr: the array to bin
    :param factor: the factor by which to bin (both the number of rows and columns in 'arr' should be integer multiples
                    of 'factor'
    :return: the binned array
    """
    shapefinal = (int(arr.shape[0] / factor), int(arr.shape[1] / factor))
    shape = (shapefinal[0], arr.shape[0] // shapefinal[0], shapefinal[1], arr.shape[1] // shapefinal[1])
    return arr.reshape(shape).sum(-1).sum(1)


def sunflowerSeedLattice(n, polar=False):
    """ Returns the coordinates of a sunflower seed lattice
    (https://demonstrations.wolfram.com/SunflowerSeedArrangements/) on a unit circle.
    If the polar argument is false, this function returns the x and y coordinates of the points as an x, y tuple,
    otherwise it returns the radius and theta coordinates as an r, theta tuple.

    :param n: the number of points in the lattice.
    :param polar: whether the coordinates should be polar (otherwise they will be cartesian)
    :return: tuple of the coordinates (either (x, y) or (r, theta))
    """
    kTheta = math.pi * 0.76393
    angle = np.linspace(kTheta, kTheta * n, n)
    r = np.linspace(0, 1, n) ** 0.5
    if polar:
        return r, angle
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y


def fibonacciLattice(n) -> Tuple[FloatOrNP, FloatOrNP, FloatOrNP]:
    """ Returns the x, y, and z coordinates of a fibonacci lattice of points on a unit sphere's surface
    (see https://arxiv.org/pdf/0912.4540.pdf).
    Points in a fibonacci lattice are very close to being evenly spaced over the sphere's surface.

    :param n: the number of points in the lattice (for best results this should be odd).
    :return: tuple of x, y, and z coordinates of the n points in the lattice.
    """
    gr = 0.5 * (1 + math.sqrt(5))
    v = 0.5 * (n - 1)
    i = np.linspace(-v, v, n)
    lat = np.arcsin(2 * i / n)
    long = 2 * math.pi * i / gr
    polar = 0.5 * np.pi - lat
    return coremaths.frame.Frame.fromSpherical(1, long, polar).tuple
