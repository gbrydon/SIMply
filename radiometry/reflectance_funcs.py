# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Bidirectional reflectance distribution functions"""

import radiometry.radiometry
from coremaths.vector import Vec3
from rendering import textures as tx
import numpy as np
import math
from typing import Callable, Dict, Optional, Tuple, Type, Union

_fnp = Union[float, np.ndarray]


class BRDF:
    """Base class for a bidirectional reflectance distribution function (BRDF).

    The BRDF of a surface gives the ratio of its reflected radiance (in a given direction) to its incident irradiance
    (from a given direction).

    Do not use instances of this base class, only use instances of fully-implemented subclasses.

    If needing to represent a BRDF with spatially-varying properties, see TexturedBRDF class.
    """
    def __init__(self):
        pass

    @staticmethod
    def lambert(dhr: _fnp) -> 'BRDFLambert':
        """Returns a lambertian surface BRDF with the given directional hemispherical reflectance
        (equivalent to normal albedo)"""
        return BRDFLambert(dhr)

    @staticmethod
    def phong(kd: _fnp, ks: _fnp, spec: _fnp) -> 'BRDFPhong':
        """ Returns a phong surface BRDF

        :param kd: diffuse reflection coefficient
        :param ks: specular reflection coefficient
        :param spec: specular exponent (the higher the value, the more the specular reflection is constrained to the
                    direction of perfect specular reflection)
        :return: phong BRDF
        """
        return BRDFPhong(kd, ks, spec)

    @staticmethod
    def hapke(w: _fnp, b: _fnp, c: _fnp, bs0: _fnp, hs: _fnp, bc0: _fnp, hc: _fnp, theta: _fnp, phi: _fnp):
        """ Initialises a new Hapke BRDF from the given parameters.

        The Hapke BRDF uses the Henyey-Greenstein single particle phase function. Both the single-term and two-term
        versions of this phase function are supported. For single-term, -1<=b<=1 and c=-1; for two-term: 0<=b<=1 and
        -1<=c<=1. See radiometry.PhaseFunctions class for more details.

        Consistent with modern Hapke model (https://doi.org/10.1017/CBO9781139025683) and classic Hapke model
        (https://doi.org/10.1016/0019-1035(84)90054-X).

        :param w: single scattering albedo
        :param b: single particle scattering asymmetry parameter (-1->1 for single-term, 0->1 for two-term)
        :param c: single particle scattering partition parameter (=-1 for single-term, -1->1 for two-term)
        :param bs0: shadow hiding opposition effect amplitude term
        :param hs: shadow hiding opposition effect width term
        :param bc0: coherent backscatter opposition effect amplitude term (for classic Hapke, set to 0)
        :param hc: coherent backscatter opposition effect width term (for classic Hapke, set to 1)
        :param theta: surface roughness parameter [radians]
        :param phi: filling factor (equal to 1 - porosity) (for classic Hapke set to 0)
        """
        return BRDFHapke(w, b, c, bs0, hs, bc0, hc, theta, phi)

    @staticmethod
    def radianceFactor(rf: _fnp) -> 'BRDFRadianceFactor':
        """ Returns a BRDF for a surface to exhibit a given radiance factor (I/F).

        This class should be used for calculating the observed radiance from a surface under observation conditions for
        which the observed I/F is known. Using this class to calculate radiances of a surface under other viewing
        conditions will yield meaningless values.

        :param rf: the radiance factor
        :return: the BRDF
        """
        return BRDFRadianceFactor(rf)

    @staticmethod
    def solarArray() -> 'BRDF':
        """A phong model approximation of a solar array spacecraft surface"""
        # return BRDFPhong(0.08, 0.02, 10)
        return BRDFPhong(0.15, 0.25, 0.26)

    @staticmethod
    def aluminium() -> 'BRDF':
        """A phong model approximation of an aluminium spacecraft surface (source and validity unknown)"""
        return BRDFPhong(0.4, 0.2, 10)

    def evaluate(self, n: 'Vec3', ls: 'Vec3', v: 'Vec3') -> _fnp:
        """ Calculates the value of this BRDF under the given observation geometry.
        The returned value is positive if
        the projection of ls onto n is parallel with n (whereas it is negative if the projection of ls onto n is
        antiparallel to n). I.e. the value of the BRDF is returned as negative if the surface is illuminated on its back
        face (the side which faces the opposite direction to surface normal n). If overriding this function from your
        own subclass, this same sign convention must be followed.

        :param n: the surface's normal vector
        :param ls: normalised vector pointing from point of reflection on surface to light source
        :param v: normalised vector pointing from point of reflection on surface to observer
        :return: the BRDF for the given viewing geometry [sr^-1]
        """
        raise NotImplemented


class BRDFLambert(BRDF):
    """Bidirectional reflectance distribution function (BRDF) for a Lambertian surface"""
    def __init__(self, dhr: _fnp):
        """ Initialises a new Lambertian surface BRDF

        :param dhr: The directional hemispherical reflectance (equal to its normal albedo) of the surface
        """
        self.dhr = dhr
        super().__init__()

    @property
    def value(self):
        """ The value of this Lambertian BRDF (a lambertian surface has a constant BRDF, independent of observation
        geometry)
        """
        return self.dhr / math.pi

    def evaluate(self, n, ls, v):
        sign = n.dot(ls) / np.abs(n.dot(ls))  # this term negates the returned value in cases of back face illumination
        if v.isNumpyType and n.isNumpyType is False and ls.isNumpyType is False:
            sign = sign * v.norm.length  # this is done to get the return value to match v's numpy shape
        return self.value * sign


class BRDFPhong(BRDF):
    """Bidirectional reflectance distribution function (BRDF) for a Phong surface"""
    def __init__(self, kd: _fnp, ks: _fnp, spec: _fnp):
        """ Initialises a new Phong surface BRDF.

        :param kd: The surface's diffuse reflection coefficient
        :param ks: The surface's specular reflection coefficient
        :param spec: The surface's specular exponent (the higher the value, the more the specular reflection is
                    constrained to the direction of perfect specular reflection)
        """
        self.kd = kd
        self.ks = ks
        self.spec = spec
        super().__init__()

    def evaluate(self, n, ls, v):
        sign = n.dot(ls) / np.abs(n.dot(ls))  # this term negates the returned value in cases of back face illumination
        r = (2 * n.dot(ls) * n) - ls  # direction of perfect specular reflection
        alpha = r.angleWith(v)
        alpha = np.where(alpha > 0.5 * np.pi, 0.5 * np.pi, alpha)
        val = (self.kd / math.pi) + (self.ks * (self.spec + 2) * (np.cos(alpha) ** self.spec) / (2 * math.pi))
        return sign * np.abs(val)


class BRDFHapke(BRDF):
    """ Hapke model BRDF (using Henyey-Greenstein single particle phase function)

    From Hapke's 2012 book https://doi.org/10.1017/CBO9781139025683"""
    def __init__(self, w: _fnp, b: _fnp, c: _fnp, bs0: _fnp, hs: _fnp, bc0: _fnp, hc: _fnp, theta: _fnp, phi: _fnp):
        """ Initialises a new Hapke BRDF form the given parameters.

        The Hapke BRDF uses the Henyey-Greenstein (HG) single particle phase function. Both the single-term and two-term
        versions of this phase function are supported. For single-term, -1<=b<=1 and c=-1; for two-term: 0<=b<=1 and
        -1<=c<=1. NOTE: the HG form used is that of https://doi.org/10.1017/CBO9781139025683 (pg. 105 eq 6.7a).
        The HG function is sometimes used in an alternative form, as in https://doi.org/10.1016/j.icarus.2018.04.001,
        in which case the value passed here for c should be equal to 1 - 2C where C is the value of c appropriate for
        this alternative form.

        Consistent with both modern Hapke model (https://doi.org/10.1017/CBO9781139025683) and classic Hapke model
        (https://doi.org/10.1016/0019-1035(84)90054-X).

        :param w: single scattering albedo
        :param b: single particle scattering lobe shape parameter (-1->1 for single-term, 0->1 for two-term)
        :param c: single particle scattering partition parameter (=-1 for single-term, -1->1 for two-term)
        :param bs0: shadow hiding opposition effect amplitude term
        :param hs: shadow hiding opposition effect width term
        :param bc0: coherent backscatter opposition effect amplitude term (for classic Hapke, set to 0)
        :param hc: coherent backscatter opposition effect width term (for classic Hapke, set to 1)
        :param theta: surface roughness parameter [radians]
        :param phi: filling factor (equal to 1 - porosity) (for classic Hapke set to 0)
        """
        self.w = w  # single scattering albedo
        self.b = b  # scattering asymmetry parameter
        self.c = c  # scattering partition parameter
        self.Bs0 = bs0  # shadow hiding opposition effect amplitude term
        self.hs = hs  # shadow hiding opposition effect width term
        self.Bc0 = bc0  # coherent backscatter opposition effect amplitude term
        self.hc = hc  # coherent backscatter opposition effect width term
        self.theta = theta  # surface roughness parameter
        self.phi = phi  # filling factor (equal to 1 - porosity)
        super().__init__()

    def evaluate(self, n: 'Vec3', ls: 'Vec3', v: 'Vec3') -> _fnp:
        # evaluates the 'modern' Hapke BRDF as described in Hapke's 2012 book: https://doi.org/10.1017/CBO9781139025683
        # page numbers are given beside some functions below for easy reference to their source in the book
        pa = ls.angleWith(v)  # phase angle
        i = ls.angleWith(n)  # incident angle
        backIlluminated = i > 0.5 * np.pi
        try:
            i[backIlluminated] = np.pi - i[backIlluminated]
        except TypeError:
            if backIlluminated:
                i = np.pi - i
        e = v.angleWith(n)  # emergent angle
        backViewed = e > 0.5 * np.pi
        try:
            e[backViewed] = np.pi - e[backViewed]
        except TypeError:
            if backViewed:
                e = np.pi - e
        psi_arg = (np.cos(pa) - np.cos(i) * np.cos(e)) / (np.sin(i) * np.sin(e))
        try:
            psi_arg[e == 0] = 0
        except IndexError:
            psi_arg[:, e == 0] = 0
        except TypeError:
            if e == 0:
                psi_arg = 0

        try:
            psi_arg[psi_arg > 1] = 1  # floating point precision error may give psi_arg > 1, so clip to 1
            psi_arg[psi_arg < -1] = -1  # floating point precision error may give psi_arg < -1, so clip to -1
        except TypeError:
            if psi_arg > 1:
                psi_arg = 1
            elif psi_arg < -1:
                psi_arg = -1
        psi = np.arccos(psi_arg)  # azimuthal angle

        u0 = np.cos(i)
        u = np.cos(e)

        chi = 1 / np.sqrt(1 + np.pi * np.tan(self.theta) ** 2)

        def E1(_a):
            return np.exp(-2 / (np.pi * np.tan(self.theta) * np.tan(_a)))

        def E2(_a):
            return np.exp(-1 / (np.pi * np.tan(self.theta) ** 2 * np.tan(_a) ** 2))

        def get_u0e():
            def uoeA():
                # for i <= e
                num = np.cos(psi) * E2(e) + np.sin(0.5 * psi) ** 2 * E2(i)
                denom = 2 - E1(e) - (psi / np.pi) * E1(i)
                return chi * (u0 + np.sin(i) * np.tan(self.theta) * num / denom)

            def uoeB():
                # for e < i
                num = E2(i) - np.sin(0.5 * psi) ** 2 * E2(e)
                denom = 2 - E1(i) - (psi / np.pi) * E1(e)
                return chi * (u0 + np.sin(i) * np.tan(self.theta) * num / denom)

            ret = uoeA()
            try:
                ret[e < i] = uoeB()[e < i]
            except TypeError:
                if e < i:
                    ret = uoeB()
            return ret

        def get_ue():
            def ueA():
                # for i <= e
                num = E2(e) - np.sin(0.5 * psi) ** 2 * E2(i)
                denom = 2 - E1(e) - (psi / np.pi) * E1(i)
                return chi * (u + np.sin(e) * np.tan(self.theta) * num / denom)

            def ueB():
                # for e < i
                num = np.cos(psi) * E2(i) + np.sin(0.5 * psi) ** 2 * E2(e)
                denom = 2 - E1(i) - (psi / np.pi) * E1(e)
                return chi * (u + np.sin(e) * np.tan(self.theta) * num / denom)

            ret = ueA()
            try:
                ret[e < i] = ueB()[e < i]
            except TypeError:
                if e < i:
                    ret = ueB()
            return ret

        u0e = get_u0e()
        ue = get_ue()

        L = u0e / (u0e + ue)

        P = radiometry.radiometry.PhaseFunctions.henyeyGreenstein(pa, self.b, self.c)

        Bs = self.Bs0 / (1 + np.tan(0.5 * pa) / self.hs)

        def get_K():
            # pg 217
            if type(self.phi) is np.ndarray:
                _x = 1.209 * self.phi ** 0.66667
                ret = -np.log(1 - _x) / _x
                ret[self.phi == 0] = 1
                return ret
            else:
                if self.phi == 0:
                    return 1
            _x = 1.209 * self.phi ** 0.66667
            return -np.log(1 - _x) / _x
        K = get_K()

        # pg 244 eqn 9.43
        Bc = self.Bc0 / (1 + (1.3 + K) * ((np.tan(0.5 * pa) / self.hc) + (np.tan(0.5 * pa) / self.hc) ** 2))

        def S():
            """Calculates the Hapke model's S term, which incorporates the effects of large scale surface roughness"""
            # eqns 12.50 (and associated) and 12.54 (and associated); pgs 321, 322
            def eta(_a):
                return chi * (np.cos(_a) + np.sin(_a) * np.tan(self.theta) * E2(_a) / (2 - E1(_a)))

            f = np.exp(-2 * np.tan(0.5 * psi))

            def SA():
                # for i <= e
                num = ue * u0 * chi
                denom = eta(e) * eta(i) * (1 - f + f * chi * (u0 / eta(i)))
                return num / denom

            def SB():
                # for e < i
                num = ue * u0 * chi
                denom = eta(e) * eta(i) * (1 - f + f * chi * (u / eta(e)))
                return num / denom

            ret = SA()
            try:
                ret[e < i] = SB()[e < i]
            except TypeError:
                if e < i:
                    ret = SB()
            return ret

        def M():
            # pg 204 eqn 8.56
            y = np.sqrt(1 - self.w)
            r0 = (1 - y) / (1 + y)

            def H(_x):
                # pg 204 eqn 8.56
                return 1 / (1 - self.w * _x * (r0 + 0.5 * (1 - 2 * r0 * _x) * np.log((1 + _x) / _x)))

            return H(u0e / K) * H(ue / K) - 1

        r = 0.25 * self.w * L * K / np.pi * ((1 + Bs) * P + M()) * (1 + Bc) * S()  # pg 323 eqn 12.55
        try:
            r[backIlluminated] = -1 * r[backIlluminated]
        except TypeError:
            if backIlluminated:
                r = -r
        return r / u0


class BRDFRadianceFactor(BRDF):
    """BRDF for a surface to exhibit a given radiance factor (I/F). This class should be used for calculating the
    observed radiance from a surface under observation conditions for which the observed I/F is known."""
    def __init__(self, rf: _fnp):
        self._rf = rf
        super().__init__()

    @property
    def rf(self):
        """The radiance factor value(s) that this BRDF exhibits."""
        return self._rf

    def evaluate(self, n: 'Vec3', ls: 'Vec3', v: 'Vec3') -> _fnp:
        return self._rf / (math.pi * n.dot(ls))


class TexturedBRDF:
    """Class for a textured BRDF.

    A textured BRDF has one or more of its parameters represented by a Texture object, and should be used when spatial
    variation of a BRDF over an object's surface is required.
    """
    def __init__(self, brdf: Union[Type[BRDF], Callable[[...], BRDF]], params: Tuple[Union[float, tx.textureType], ...]):
        """ Initialises a new textured BRDF.

        :param brdf: either 1) the BRDF class (NOT an instance of the class) associated with this textured BRDF (e.g.
            BRDFLambert), or 2) a function which takes some number of parameters and returns an instance of the BRDF
            associated with this textured BRDF (e.g. BRDF.lambert).
        :param params: a tuple of the BRDF's parameter(s), in the order that the BRDF's initialiser takes them as
            arguments. The paramaters can each be either a float or a Texture object. If multiple parameters are a
            Texture object, they must be of the same texture type.
        """
        self._brdfClass = brdf
        self._params = params
        self._isTextured = False
        self._textureType = None
        for param in self._params:
            if type(param) is not float and type(param) is not int and type(param) is not np.float64:
                self._isTextured = True
                if self._textureType is not None:
                    message = "All textured parameters within a single textured BRDF must be of the same texture type."
                    assert type(param) is self._textureType, message
                else:
                    self._textureType = type(param)

    @property
    def isTextured(self):
        """ Whether this textured BRDF is textured (it is textured if at least one of its parameters is a Texture
        object). If this is false, this textured BRDF acts the same as a standard BRDF (consider using a standard BRDF
        class where no textured parameters are required).

        :return: whether this BRDF is textured.
        """
        return self._isTextured

    @property
    def textureType(self):
        """The type of texture that this brdf's textured parameters are represented by (all textured parameters of a
        textured brdf must use the same texture type)"""
        return self._textureType

    def brdf(self, tc: Union[Tuple[_fnp, _fnp], Vec3]) -> BRDF:
        """ Returns a BRDF for the given surface location defined by the given texture coordinates.

        The returned BRDF can then be evaluated to calculate the reflected radiance at the given location.

        :param tc: Texture coordinates (must be either a (u,v) tuple or a Vec3 position (the latter being for
            planetocentric textures))
        :return: a BRDF for the given location
        """
        params = []
        for param in self._params:
            if type(param) is float or type(param) is int or type(param) is np.float64:
                # the parameter is a single value (not a texture)
                params += [param]
                continue
            # the parameter is a texture, so its value(s) must be retrieved using texture coordinates
            if type(tc) is Vec3:
                params += [param.valueFromXYZ(tc)]
            else:
                u, v = tc
                params += [param.valueFromUV(u, v)]
        return self._brdfClass(*params)


class SpectralBRDF:
    """Class for spectral BRDFs.

    The spectral BRDF class should be used for representing BRDFs with spectral dependence (i.e. different values
    at different wavelengths).

    A spectral BRDF object contains multiple BRDFs (either standard or textured BRDFs), each with an associated
    wavelength value at which that BRDF is valid (stored in a dictionary with wavelength as key).

    A spectral BRDF can also have a default BRDF,
    which is used for all wavelengths which are not explicitly assigned a BRDF."""

    def __init__(self, brdfs: Dict[Optional[float], Union[BRDF, TexturedBRDF]]):
        """ Initialises a new spectral BRDF. The dictionary passed for brdfs should contain BRDFs (either standard or
        textured) for desired wavelengths (with the wavelengths as their keys).

        Include a BRDF with a key of None if a default BRDF (i.e. one that applies to all wavelengths for which a
        specific BRDF is not provided) is desired.

        :param brdfs: dictionary of individual wavelengths and their associated BRDFs (either standard or textured).
        """
        self._brdfs = brdfs

    @property
    def textureType(self):
        """If this spectral BRDF contains any textured BRDFs, this returns their type (they must all be of the same
        type), otherwise it returns None."""
        for brdf in self._brdfs.values():
            if brdf.isTextured:
                return brdf.textureType
        return None

    @property
    def defaultBRDF(self):
        """The default BRDF for wavelengths for which there is not an assigned BRDF. If no default BRDF has been
        assigned to this spectral BRDF, this returns None."""
        try:
            return self._brdfs[None]
        except KeyError:
            return None

    def add(self, brdf: Union[BRDF, TexturedBRDF], wavelength: Optional[float]):
        """ Adds a new BRDF to this spectral BRDF, for the given wavelength.

        :param brdf: BRDF to add
        :param wavelength: wavelength [nm] at which the BRDF applies
        """
        self._brdfs[wavelength] = brdf

    def atWavelength(self, w: float) -> Optional[Union[BRDF, TexturedBRDF]]:
        """ This spectral BRDF's BRDF at the given wavelength. If this spectral BRDF has no BRDF for the given
        wavelength, its default BRDF will instead be returned, or None if there is no default BRDF.

        :param w: wavelength at which the BRDF is required.
        :return: BRDF if one is available, or None.
        """
        try:
            return self._brdfs[w]
        except KeyError:
            try:
                return self._brdfs[None]
            except KeyError:
                return None
