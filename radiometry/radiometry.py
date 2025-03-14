# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Functions and classes for physically accurate radiometric modelling and calculations."""

import os
from simply_utils import paths, constants as consts
from coremaths import geometry as gm
from coremaths.vector import Vec3
from radiometry.reflectance_funcs import BRDF
import math
import numpy as np
from scipy import interpolate
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Optional, List, Callable

_fnp = Union[float, np.ndarray]
_inp = Union[int, np.ndarray]
_poly = Union[List[gm.Polygon], gm.Polyhedron]


class PhaseFunctions:
    """Class containing reflection/scattering phase functions"""
    @staticmethod
    def lambertSphere(ang: _fnp) -> _fnp:
        """Evaluates the phase function of a uniform lambertian sphere at the given phase angle [radians]

        :param ang: the phase angle [radians]
        :return: the value of the phase function at the given phase angle
        """
        return ((math.pi - ang) * np.cos(ang) + np.sin(ang)) / math.pi

    @staticmethod
    def henyeyGreenstein(ang: _fnp, b: _fnp, c: _fnp) -> _fnp:
        """ Evaluates the Henyey-Greenstein (HG) single particle phase function at the given phase angle [radians].

        Can be used for two-term and single-term HG phase function.

        This phase function takes the form described in https://doi.org/10.1017/CBO9781139025683 (pg. 105 eq 6.7a).
        The Henyey-Greenstein function is sometimes used in the alternative form given in
        https://doi.org/10.1016/j.icarus.2018.04.001, in which case the value passed to this function for c should be
        equal to 1 - 2C where C is the value of c if using the alternative form.

        For two-term: 0<=b<=1, -1<=c<=1

        For single-term: -1<=b<=1, c=-1

        :param ang: the phase angle [radians]
        :param b: amplitude of forward and backward scattering lobes (peaks)
        :param c: two-term function's relative lobe strength (for single-term HG, set to -1)
        :return: the value of the phase function at the given phase angle
        """
        P1 = 0.5 * (1 + c) * (1 - b ** 2) / (1 - 2 * b * np.cos(ang) + b ** 2) ** 1.5
        P2 = 0.5 * (1 - c) * (1 - b ** 2) / (1 + 2 * b * np.cos(ang) + b ** 2) ** 1.5
        return P1 + P2


def photonEnergy(wavelength: _fnp) -> _fnp:
    """ Calculates the energy of a photon of given wavelength

    :param wavelength: the photon's wavelength [nm]
    :return: the photon's energy [J]
    """
    return consts.hPlanck * consts.cLight / (wavelength * 1e-9)


def photonWavelength(energy: _fnp) -> _fnp:
    """ Calculates the wavelength of a photon of given energy

    :param energy: the photon's energy [J]
    :return: the photon's wavelength [nm]
    """
    return 1e9 * consts.hPlanck * consts.cLight / energy


def janskyToSI(flux, wavelength):
    """ Converts a spectral flux from Janskys to SI units (W m^-2 nm^-1)

    :param flux: the flux in Janskys
    :param wavelength: the wavelength associated with the flux value [nm]
    :return: the flux in SI units [W m^-2 nm^-1]
    """
    hzpernm = 3e8 * 1e-9 / (wavelength * 1e-9) ** 2
    return flux * 1e-26 * hzpernm


def janskyFromSI(flux, wavelength):
    """ Converts a spectral flux from SI units (W m^-2 nm^-1) to Janskys

    :param flux: the flux in SI units [W m^-2 nm^-1]
    :param wavelength: the wavelength associated with the flux value [nm]
    :return: the flux in Janskys
    """
    frequency = 3e8 / (wavelength * 1e-9)
    nmperhz = 3e8 * 1e9 / frequency ** 2
    return flux * nmperhz / 1e-26


def fluxFromMag(mag, f0):
    """ Converts an apparent magnitude to flux

    :param mag: the magnitude
    :param f0: the flux corresponding to a magnitude of zero
    :return: the flux, in the same units as the zero flux
    """
    return f0 * 10 ** (-mag / 2.5)


def fluxToMag(flux, f0):
    """ Converts a flux to an apparent magnitude

    :param flux: the flux (in the same units as f0)
    :param f0: the flux corresponding to a magnitude of 0
    :return: the magnitude
    """
    return -2.5 * np.log10(flux / f0)


def fluxFromIntensity(i: _fnp, r: _fnp) -> _fnp:
    """Returns the flux [W m^-2] observed a distance r [m] from a point source that is reflecting/emitting light with an
    intensity i [W sr^-1] towards the observer

    :param i: the light's intensity in the direction of the observer [W sr^-1]
    :param r: distance between point source object and observer [m]
    :return: the flux observed [W m^-2]
    """
    return i / (r ** 2)


def fluxToIntensity(f: _fnp, r: _fnp) -> _fnp:
    """Returns the intensity [W sr^-1] that an object is reflecting/emitting toward an observer a distance r [m] who is
    observing a flux f [W m^-2]

    :param f: observed flux [W m^-2]
    :param r: distance between point source object and observer [m]
    :return: the intensity being reflected/emitted toward the observer [W sr^-1]
    """
    return f * r ** 2


def solarFluxBetween(lower: float, upper: float) -> float:
    """Returns the total solar flux [W m^-2] at 1AU between the two given wavelengths (inclusive).
    Calculated from 2000 ASTM Standard Extraterrestrial Spectrum Reference E-490
    (see https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html). Only valid for wavelengths from 200-10000 nm.

    :param lower: lower bound of wavelength range [nm]
    :param upper: upper bound of wavelength range [nm]
    :return: total solar flux at 1AU within given wavelength range [W m^-2]
    """
    path = '../resources/spectra/solar_spectrum_1AU_raw.xlsx'
    df = pd.read_excel(path)
    w = df['wavelength [nm]'].to_numpy()
    if lower < np.nanmin(w) or upper > np.nanmax(w):
        print("Warning: you have requested a lower and/or upper wavelength limit outside of the data's range. "
              "You requested solar flux between {0} nm and {1} nm, but the data are limited to the range "
              "{2} nm - {3} nm. Trying to calculate outside this range will give inaccurate "
              "results".format(lower, upper, np.nanmin(w), np.nanmax(w)))
    fluxcont = df["row's flux contribution [W/m2]"].to_numpy()
    return np.nansum(fluxcont[(lower <= w) * (w <= upper)])


def lambertSurfaceRadiance(irr, dhr):
    """ Calculates the reflected radiance from a point on a Lambertian surface

    :param irr: normal irradiance [W m^-2] of the surface at the point
    :param dhr: the directional hemispherical reflectance of the surface (equal to its normal albedo)
    :return: the radiance of the Lambertian surface [W m^-2 sr^-1]
    """
    brdf = BRDF.lambert(dhr)
    return irr * brdf.value


def lambertElementIntensity(irr, dhr, sa, angr):
    """ Calculates the total intensity reflected from a Lambertian surface element of given surface area along a given
    angle of reflection. This function assumes angle of incidence and angle of reflection to be uniform over whole
    surface element - this is a good approximation for surface-light and surface-observer separations that are
    significantly larger than the surface element's length.

    :param irr: normal irradiance [W m^-2] of the surface element
    :param dhr: the directional hemispherical reflectance of the surface (equal to its normal albedo)
    :param sa: surface area of the surface element [m^2]
    :param angr: the angle of reflection (measured from the surface normal) [radians]
    :return: the Lambertian surface element's intensity [W sr^-1]
    """
    return sa * np.cos(angr) * lambertSurfaceRadiance(irr, dhr)


def lambertSphereIntensity(f: _fnp, ang: _fnp, r: _fnp, albedo: _fnp, bond=True) -> _fnp:
    """Returns the total intensity [W sr^-1] reflected from a uniform lambertian sphere with given properties.

    :param f: flux incident on the sphere [W m^-2]
    :param ang: phase angle of reflection [radians]
    :param r: radius of sphere [m]
    :param albedo: albedo of the sphere (geometric or bond albedo, as specified by the 'bond' argument
    :param bond: whether the supplied albedo is the bond albedo (=True) or the geometric albedo (=False)
    :return: the reflected intensity [W sr^-1]
    """
    if bond:
        albedo = 2 * albedo / 3
    return albedo * r ** 2 * f * PhaseFunctions.lambertSphere(ang)


def surfaceRadiance(f: _fnp, n: 'Vec3', ls: 'Vec3', v: 'Vec3', brdf: Union[BRDF, _fnp]) -> _fnp:
    """ The radiance of light reflected from a surface of given BRDF

    :param f: the flux incident on the surface [W m^-2]
    :param n: the surface's normal vector
    :param ls: normalised vector pointing from point of reflection on surface to light source
    :param v: normalised vector pointing from point of reflection on surface to observer
    :param brdf: the surface's BRDF as a BRDF instance (which will be evaluated by this function), or as a float/ndarray
        giving the result of the BRDF already having been evaluated for the given reflection geometry.
    :return: the surface's radiance [W m^-2 sr^-1]
    """
    irr = f * n.dot(ls)  # the normal irradiance of the surface
    if isinstance(brdf, BRDF):
        # brdf is an instance of a BRDF class and needs evaluating for the given reflection geometry to get the ratio
        # of reflected radiance to incident irradiance
        return irr * brdf.evaluate(n, ls, v)
    else:
        # brdf is the ratio of reflected radiance to incident irradiance (i.e. brdf has been evaluated before being
        # passed to this function)
        return irr * brdf


def surfaceElementIntensity(f: _fnp, sa: _fnp, n: 'Vec3', ls: 'Vec3', v: 'Vec3', brdf: 'BRDF') -> _fnp:
    """ Calculates the total intensity reflected from a surface element of given BRDF. This function assumes angle of
    incidence and angle of reflection to be uniform over whole surface element - this is a good approximation for
    surface-light and surface-observer separations that are significantly larger than the surface element's length.

    :param f: the flux incident on the disc [W m^-2]
    :param sa: surface area of the surface element [m^2]
    :param n: the surface's normal vector
    :param ls: normalised vector pointing from point of reflection on surface to light source
    :param v: normalised vector pointing from point of reflection on surface to observer
    :param brdf: the surface's BRDF
    :return: the surface element's intensity [W sr^-1]
    """
    proj = sa * n.dot(v)  # the projected area as seen by the observer
    return proj * surfaceRadiance(f, n, ls, v, brdf)


def convexPolyIntensity(poly: _poly, brdf: Union[BRDF, List[BRDF]], f: _fnp, ls: Vec3, v: Vec3) -> _fnp:
    """ Calculates the total intensity reflected by the given polygon or convex polyhedron under the given illumination
    and observation geometry. The intensity is calculated under the assumption that the light source and observer
    are far from the object (such that incident flux, angle of reflection and angle of incidence are constant over
    the whole of each polygon surface of the object).

    If the object is a polyhedron, it must be a convex polyhedron (self-shadowing of
    a concave polygon will not be properly calculated, leading to overestimation of intensity).

    If the object is a polygon, only its positive face (the side facing the +ve direction of the surface normal) will
    be treated as being able to reflect flux.

    :param poly: the convex polyhedron's geometry (either a list of polygons or a convex polyhedron object)
    :param brdf: either a single BRDF describing the reflection properties of all faces of the object, or a list of
        BRDFs, one for each face of the polyhedron, in the same order as those faces appear in the polyhedron's faces
        list. Only spatially-uniform BRDFs are supported.
    :param f: the flux incident on the object [W m^-2]
    :param ls: normalised vector pointing from the object's centre to the illumination source
    :param v: normalised vector pointing from the object's centre to the observer
    :return: the total intensity reflected toward the observer by the object [W sr^-1]
    """
    if type(poly) is list:
        faces = poly
    else:
        faces = poly.faces
    total = 0
    for n, face in enumerate(faces):
        frame = face.frame
        if type(brdf) is list:
            faceBRDF = brdf[n]
        else:
            faceBRDF = brdf
        illuminated = frame.fromWorld(frame.origin + ls).z > 0
        visible = frame.fromWorld(v + frame.origin).z > 0
        intensity = surfaceElementIntensity(f * illuminated, face.area, frame.w, ls, v, faceBRDF) * visible
        if type(intensity) is np.ndarray:
            intensity[intensity < 0] = 0
        elif intensity < 0:
            intensity = 0
        total += intensity
    return total


class SpectralDensityCurve:
    """A class representing a spectral density curve with units of [... nm^-1]"""
    __array_priority__ = 1000

    def __init__(self, values: np.ndarray, wavelengths: np.ndarray):
        """ Initialises a new spectral density curve

        :param values: the values of the curve [nm^-1]
        :param wavelengths: the wavelengths of the curve [nm]
        """
        self._values = values  # [... nm^-1]
        self._wavelengths = wavelengths  # [nm]

    @staticmethod
    def solarSpectrum1AU():
        """Returns a spectrum of the solar flux density [W m^-2 nm^-1] at 1 AU from the Sun over the
        200-10000nm wavelength range"""
        dirPath = os.path.join(paths.dataDirPath(), "input", "solar_flux_den_1AU")
        wavelengths = np.load(os.path.join(dirPath, "wavelength.npy"))
        fluxDens = np.load(os.path.join(dirPath, "spectral_flux_density.npy"))
        return SpectralDensityCurve(fluxDens, wavelengths)

    @classmethod
    def uniform(cls, value: float, wavelengths: np.ndarray):
        """Returns a spectral density curve with spectrally uniform value

        :param value: the uniform value [nm^-1]
        :param wavelengths: the wavelengths of the spectral density curve [nm]
        :return: the spectral density curve
        """
        values = np.full_like(wavelengths, value)
        return cls(values, wavelengths)

    @classmethod
    def fromExpression(cls, expression: Callable[[np.ndarray], np.ndarray], wavelengths: np.ndarray):
        """Returns a spectral density curve whose values are calculated from a given expression at the given wavelengths

        :param expression: expression that takes wavelength [nm] and returns a spectral density value [nm^-1]
        :param wavelengths: the wavelengths at which values are calculated
        :return: the spectral density curve
        """
        curve = expression(wavelengths)
        return cls(curve, wavelengths)

    @property
    def values(self) -> np.ndarray:
        """This curve's values [... nm^-1]"""
        return self._values

    @property
    def wavelengths(self) -> np.ndarray:
        """This curve's wavelengths [nm]"""
        return self._wavelengths

    @property
    def normalised(self) -> 'SpectralDensityCurve':
        """Returns this curve normalised to its maximum value"""
        return self.__init__(self._values / np.max(self._values), wavelengths=self._wavelengths)

    @property
    def fromPowerToPhotonRate(self) -> 'SpectralDensityCurve':
        """If this spectral density curve is a spectrum of radiated power [W nm^-1] or flux [W m^-2 nm^-1], this
        calculates the spectral photon rate [photons s^-1 nm^-1] or [photons s^-1 m-2 nm^-1] and returns it as a
        spectral density curve."""
        photonPerEnergy = 1 / photonEnergy(self._wavelengths)  # photons J^-1
        return self * photonPerEnergy

    @property
    def fromPhotonRateToPower(self) -> 'SpectralDensityCurve':
        """If this spectral density curve is a spectrum of radiated photon rate [photons s^-1 nm^-1] or
        [photons s^-1 m-2 nm^-1], this calculates the radiated power [W nm^-1] or flux [W m^-2 nm^-1] and returns it as
        a spectral density curve."""
        energyPerPhoton = photonEnergy(self._wavelengths)  # J photon^-1
        return self * energyPerPhoton

    def scaledTo(self, value: float, w: float):
        factor = value / self.valueAt(w)
        return factor * self

    def valueAt(self, w: float) -> Optional[float]:
        """ The spectral density curve's value at the given wavelength.

        :param w: The wavelength [nm]
        :return: The value [... nm^-1]
        """
        if w < np.nanmin(self._wavelengths):
            return None
        if w > np.nanmax(self._wavelengths):
            return None
        try:
            return float(self._values[self._wavelengths == w])
        except TypeError:
            return float(np.interp(w, self._wavelengths, self._values))

    def integrated(self, min_w: float = None, max_w: float = None) -> float:
        """ Returns the result of integrating this spectral density curve over the given wavelength range
        (if no min or max wavelengths are given the min and max (respectively) wavelength values associated with this
        spectral density curve are used as the limits of integration)

        :param min_w: minimum wavelength [nm] bound for integration (curve's minimum wavelength used if None is given)
        :param max_w: maximum wavelength [nm] bound for integration (curve's maximum wavelength used if None is given)
        :return:
        """
        if min_w is None:
            min_w = self._wavelengths[0]
        if max_w is None:
            max_w = self._wavelengths[-1]
        condition = (self._wavelengths >= min_w) * (self._wavelengths <= max_w)
        w = self._wavelengths[condition]
        wBounds = 0.5 * (w[1:] + w[:-1])
        wBounds = np.append(np.array((min_w,)), wBounds)
        wBounds = np.append(wBounds, np.array((max_w,)))
        dw = wBounds[1:] - wBounds[:-1]
        v = self._values[condition]
        return np.nansum(v * dw)

    def interpolated(self, wavelengths: np.ndarray):
        """ Creates a new spectral density curve by interpolating the values of this spectral density curve at the
        given wavelengths

        :param wavelengths: the wavelengths [nm] at which to interpolate spectral values
        :return: the interpolated spectral density curve
        """
        f = interpolate.interp1d(self._wavelengths, self._values)
        newVals = f(wavelengths)
        return SpectralDensityCurve(newVals, wavelengths)

    def display(self, xscale=None, yscale=None):
        """Displays the spectral curve, using matplotlib

        :param xscale: xscale to use for x axis, as per pyplot
        :param yscale: y scale to use for y axis, as per pyplot
        """
        plt.plot(self._wavelengths, self._values)
        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Value [... nm^-1]')
        plt.show()

    def __neg__(self):
        return SpectralDensityCurve(-self._values, self._wavelengths)

    def __add__(self, other: 'SpectralDensityCurve'):
        s = "To add two spectral density curves, they must represent values at the same wavelengths"
        assert self._wavelengths.size == other.wavelengths.size, s
        assert np.all(self._wavelengths == other.wavelengths), s
        return SpectralDensityCurve(self._values + other.values, self._wavelengths)

    def __sub__(self, other: 'SpectralDensityCurve'):
        s = "To subtract a spectral density curve from another, they must represent values at the same wavelengths"
        assert self._wavelengths.size == other.wavelengths.size, s
        assert np.all(self._wavelengths == other.wavelengths), s
        return SpectralDensityCurve(self._values - other.values, self._wavelengths)

    def __mul__(self, value: Union[float, 'SpectralDensityCurve']):
        if type(value) is SpectralDensityCurve:
            s = "To multiply two spectral density curves, they must represent values at the same wavelengths"
            assert self._wavelengths.size == value.wavelengths.size, s
            assert np.all(self._wavelengths == value.wavelengths), s
            return SpectralDensityCurve(self._values * value.values, self._wavelengths)
        return SpectralDensityCurve(self._values * value, self._wavelengths)

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, value):
        if type(value) is SpectralDensityCurve:
            s = "To divide a spectral density curve by another, they must represent values at the same wavelengths"
            assert self._wavelengths.size == value.wavelengths.size, s
            assert np.all(self._wavelengths == value.wavelengths), s
            return SpectralDensityCurve(self._values / value.values, self._wavelengths)
        return SpectralDensityCurve(self._values / value, self._wavelengths)
