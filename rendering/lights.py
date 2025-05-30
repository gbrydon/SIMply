# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module containing classes for describing light sources for rendering and radiometry"""

from coremaths import math2
from coremaths.vector import Vec3
from coremaths.ray import Ray
from radiometry.radiometry import SpectralDensityCurve
import numpy as np
from simply_utils import constants as consts
from typing import Union

_inp = Union[int, np.ndarray]
_fnp = Union[float, np.ndarray]


class Light:
    """Base class representing a light source.

    Don't use this base class, only use subclasses."""

    @staticmethod
    def pointSource(pos: 'Vec3', power: Union[float, SpectralDensityCurve]):
        """ Returns a point source light (radiates its light energy evenly into 4pi sr).

        :param pos: position of the light [m]
        :param power: spectral power output of the light [W nm^-1] as either a single value (for light with
            spectrally-uniform power output), or a spectral density curve for light with spectrally-dependent
            power output.
        :return: the light
        """
        return LightPointSource(pos, power)

    @staticmethod
    def sunPointSource(pos: 'Vec3'):
        """ Returns a point source with power output equal to the sun.

        :param pos: the position of the light [m]
        :return: the solar light source
        """
        return LightPointSource.sun(pos)

    def __init__(self, pos: Vec3):
        self.pos = pos

    def fluxDensity(self, point: Vec3, w1: float, w2: float) -> float:
        """ Returns the total flux density [W m^-2] from this light source at the given point over the given wavelength
        range.

        :param point: the point (in the same coordinate frame as this light's position) [m]
        :param w1: wavelength range's minimum [nm]
        :param w2: wavelength range's maximum [nm]
        :return: the flux density [W m^-2]
        """
        return 0

    def traceRayToCentre(self, origin: Vec3) -> Ray:
        """ Returns a ray which points from the given origin point (in world coords) to this light's centre

        :param origin: Ray's origin (in world coordinates)
        :return: Ray pointing towards this light
        """
        d = (self.pos - origin).norm
        return Ray(origin, d)

    def traceRayRandom(self, origin: Vec3) -> Ray:
        """ For lights with an extended emitting surface, this function returns a ray which points from the given origin
        point (in world coords) to a random point on the visible surface of the light (from the origin's perspective).

        If this function is called on a light with an infinitesimal
        source (e.g. a point light) the returned ray point's to the light's centre.

        :param origin: Ray's origin (in world coordinates)
        :return: Ray pointing towards a random point on this light's emitting surface
        """
        return self.traceRayToCentre(origin)

    def traceRayDistributed(self, origin: Vec3, n: _inp, n_total: int) -> Ray:
        """ For lights with an extended emitting surface, this function returns a ray which points from the given origin
        point (in world coordinates) to a point on the visible surface (from the origin's perspective) which is one of
        an ordered set of evenly distributed points. The ordered set of distributed points contains n_total points, and
        the returned ray points towards the nth point in this set.

        If this function is called on a light with an infinitesimal
        source (e.g. a point light) the returned ray points to the light's centre.

        :param origin: Ray's origin (in world coordinates)
        :param n: the index of the point within the ordered set of n_total evenly distributed points, to which the ray
            is aimed.
        :param n_total: total number of points in the ordered set of evenly distributed points.
        :return: the ray
        """
        return self.traceRayToCentre(origin)


class LightPointSource(Light):
    """Class representing a point light source which radiates its light energy evenly in all directions
    (4 pi steradians)"""
    def __init__(self, pos: 'Vec3', power: Union[float, SpectralDensityCurve]):
        """ Initialises a point source light (radiates its light energy evenly into 4pi sr).

        :param pos: position of the light source [m]
        :param power: spectral power output of the light [W nm^-1] as either a single value (for light with
            spectrally-uniform power output), or a spectral density curve for light with spectrally-dependent
            power output.
        """
        super().__init__(pos)
        self._power = power
        self._radius = 0

    @classmethod
    def withFluxDensity(cls, f: Union[float, SpectralDensityCurve], loc: Union[_fnp, Vec3], light_pos: Vec3):
        """ Returns a point light source with given spectral flux density [W m^-2 nm^-1] at the given location
        (defined either by a position, or a distance from the light source)

        :param f: flux density at the given location [W m^-2 nm^-1]
        :param loc: either Vec3 location [m] where flux = f, or distance [m] from light at which flux = f
        :param light_pos: position of the light source
        :return: the light source
        """
        if type(loc) is Vec3:
            r = (light_pos - loc).length
        else:
            r = loc
        power = 4 * np.pi * r ** 2 * f
        return cls(light_pos, power)

    @classmethod
    def sun(cls, pos: Vec3):
        """ Returns a point source with power output equal to the sun.

        :param pos: position of light source [m]
        :return: the solar light source
        """
        au = consts.au
        f = SpectralDensityCurve.solarSpectrum1AU()
        light = cls.withFluxDensity(f, au, pos)
        light.radius = consts.solarRad
        return light

    @property
    def power(self) -> Union[float, SpectralDensityCurve]:
        """The spectral power output of the light [W nm^-1].

        This property is either a single value (for a light with spectrally-independent power output) or a spectral
        density curve (for a light with spectrally-dependent power output)."""
        return self._power

    @power.setter
    def power(self, new_value: Union[float, SpectralDensityCurve]):
        """Sets the power output of this light [W nm^-1]. For a light with spectrally-uniform power output provide a
        single float value. For a light with spectrally-dependent power output, provide a spectral density curve."""
        self._power = new_value

    @property
    def radius(self):
        """Radius [m] of the light's spherical surface. Zero by default, but give a non-zero value for simulating soft
        shadows due to partial shadowing of the light source."""
        return self._radius

    @radius.setter
    def radius(self, new_value):
        """ Sets the radius of the light's spherical surface. Zero by default, but give a non-zero value for simulating
        soft shadows due to partial shadowing of the light source.

        :param new_value: the value to set the radius to. [m]
        """
        self._radius = new_value

    def fluxDensity(self, point: Vec3, w1: float, w2: float):
        r = (point - self.pos).length
        try:
            return self.power.integrated(w1, w2) / (4 * np.pi * r ** 2)
        except AttributeError:
            return self.power / (4 * np.pi * r ** 2)

    def traceRayRandom(self, origin: Vec3) -> Ray:
        principal = super().traceRayToCentre(origin)
        if self._radius == 0:
            return principal
        perp = Vec3.vectorPerpendicularTo(principal.d)
        theta = np.random.random(origin.numpyShape) * 2 * np.pi
        offset = perp.rotated(principal.d, theta) * np.sqrt(np.random.random(origin.numpyShape)) * self._radius
        d = ((self.pos + offset) - origin).norm
        return Ray(origin, d)

    def traceRayDistributed(self, origin: Vec3, n: _inp, n_total: int) -> Ray:
        principal = super().traceRayToCentre(origin)
        if self._radius == 0:
            return principal
        perp = Vec3.vectorPerpendicularTo(principal.d)
        r, theta = math2.sunflowerSeedLattice(n_total, polar=True)
        r = r[n]
        theta = theta[n]
        offset = perp.rotated(principal.d, theta) * r * self._radius
        d = ((self.pos + offset) - origin).norm
        return Ray(origin, d)
