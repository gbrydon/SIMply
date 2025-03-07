# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module containing classes for representing and working with cameras."""

import math
import cv2
import numpy as np
from coremaths import math2
from coremaths.frame import Frame
from coremaths.vector import Vec2, Vec3, Mat3
from coremaths.ray import Ray
from radiometry import radiometry as rd
from scipy.ndimage.filters import gaussian_filter
import warnings
from typing import Optional, Tuple, Union

_fnp = Union[float, np.ndarray]


class Camera:
    """Base class for a general projective camera.

    Do not use this base class directly - use subclasses of it.
    """

    def __init__(self, dc: int, dr: int):
        """ Initialises a new camera with the given number of detector columns and detector rows

        :param dc: number of columns on detector
        :param dr: number of rows on detector
        """
        self._dc = dc  # number of columns of pixels on the detector
        self._dr = dr  # number of rows of pixels on the detector
        self._frame = Frame.world()  # the camera's coordinate frame, defining its position and view direction
        self._epd = 1e-3  # the camera's entrance pupil diameter [m], initialised as 1mm by default
        # (epd value should be updated as needed if using this camera for radiometric calculations)
        self._tr = 1  # the total optical transmission of the camera, initialised as 1 by default (update as needed)
        self._psfSigma = 0  # sigma value for the camera's 2D gaussian PSF (0 by default, should be updated as needed)
        self._jd = 30  # the camera's dark current [e- s^-1], initialised as 30 by default (should be updated as needed)
        #  non-uniform dark current can be simulated by setting self.jd to a 2D numpy array with same size as detector
        self._nr = 15  # the camera's RMS read noise [e-], initialised as 15 by default (should be updated as needed)
        #  non-uniform read noise can be simulated by setting self.nr to a 2D numpy array with same size as detector
        self._binning = 1  # pixel binning factor (=1 means no binning). Binning is performed on-chip prior to readout
        self._qe = 0.5  # the camera's quantum efficiency, initialised as 0.5 by default (should be updated as needed)
        self._bitdepth = 8  # the camera's bits per pixel, initialised as 8 by default (should be updated as needed)
        self._fwc = 30000  # the camera's full well capacity [e-] (30,000 by default, should be updated as needed)

    @staticmethod
    def pinhole(fov: Tuple[float, float], dc: int, dr: int, dwx=None, dwy=None, degrees=True) -> 'Pinhole':
        """ Returns a camera with ideal pinhole imaging geometry.

        :param fov: The FOV of the camera as a (horizontal FOV, vertical FOV) tuple. If the 'degrees' argument is set to
                    True, this is taken to be degrees, otherwise is treated as radians.
        :param dc: number of columns of pixels on detector.
        :param dr: number of rows of pixels on detector.
        :param dwx: detector width (physical length of a row of pixels) [m].
        :param dwy: detector height (physical length of a column of pixels) [m].
        :param degrees: whether the fov is given in degrees (True) or radians (False).
        :return: the camera.
        """
        if degrees:
            fov = (np.radians(fov[0]), np.radians(fov[1]))
        if dwx is None:
            if dwy is None:
                dwx = 1e-2
                dwy = dwx * np.tan(0.5 * fov[1]) / np.tan(0.5 * fov[0])
            else:
                dwx = dwy * np.tan(0.5 * fov[0]) / np.tan(0.5 * fov[1])
        if dwy is None:
            dwy = dwx * np.tan(0.5 * fov[1]) / np.tan(0.5 * fov[0])
        f = 0.5 * dwx / np.tan(0.5 * fov[0])
        fy = 0.5 * dwy / np.tan(0.5 * fov[1])
        a = fy / f
        cam = Pinhole(f, dc, dr, dwx, dwy)
        cam.focalAspect = a
        return cam

    @staticmethod
    def fisheye(fov: Tuple[float, float], dc: int, dr: int, dwx=None, dwy=None, degrees=True) -> 'Fisheye':
        """ Returns a fisheye camera with the given parameters

        :param fov: The FOV of the camera as a (horizontal FOV, vertical FOV) tuple. If the 'degrees' argument is set to
                    True, this is taken to be degrees, otherwise is treated as radians.
        :param dc: number of columns of pixels on detector.
        :param dr: number of rows of pixels on detector.
        :param dwx: detector width (physical length of a row of pixels) [m].
        :param dwy: detector height (physical length of a column of pixels) [m].
        :param degrees: whether the fov is given in degrees (True) or radians (False).
        :return: the camera.
        """
        if degrees:
            fov = (np.radians(fov[0]), np.radians(fov[1]))
        if dwx is None:
            if dwy is None:
                dwx = 1e-2
                dwy = dwx * fov[1] / fov[0]
            else:
                dwx = dwy * fov[0] / fov[1]
        if dwy is None:
            dwy = dwx * fov[1] / fov[0]
        f = dwx / fov[0]
        cam = Fisheye(f, dc, dr, dwx, dwy)
        return cam

    @property
    def pos(self) -> 'Vec3':
        """The position of this camera's optical centre (i.e. centre of projection) in the world frame"""
        return self._frame.origin

    @pos.setter
    def pos(self, new: 'Vec3'):
        """Sets the camera's position (i.e. the location of its optical centre) to the given vector"""
        self._frame.origin = new

    @property
    def epd(self) -> _fnp:
        """The entrance pupil diameter of the camera [m]"""
        return self._epd

    @epd.setter
    def epd(self, new: _fnp):
        """Sets a new value for the camera's entrance pupil diameter

        :param new: new entrance pupil diameter [m]
        """
        self._epd = new

    @property
    def epa(self) -> _fnp:
        """The area [m^2] of the camera's entrance pupil"""
        return 0.25 * math.pi * self.epd ** 2

    @epa.setter
    def epa(self, new: _fnp):
        """ Updates the camera's entrance pupil diameter to give the entrance pupil the given area [m^2]

        :param new: The new area of the camera's entrance pupil [m^2]
        """
        self._epd = 2 * (new / math.pi) ** 0.5

    @property
    def frame(self) -> Frame:
        """The camera's coordinate frame. The origin of the frame defines the position of the camera's optical centre.
        The frame's axes define the camera's orientation: frame's z axis is the camera's boresight, frame's x-axis is
        aligned with the camera's horizontal image axis."""
        return self._frame

    @frame.setter
    def frame(self, new: 'Frame'):
        """Updates the camera's coordinate frame. The origin of the frame defines the position of the camera's optical centre.
        The frame's axes define the camera's orientation: frame's z axis is the camera's boresight, frame's x-axis is
        aligned with the camera's horizontal image axis.

        :param new: the new coordinate frame for the camera
        """
        self._frame = new

    @property
    def dc(self) -> int:
        """The number of columns on the detector"""
        return self._dc

    @property
    def dr(self) -> int:
        """The number of rows on the detector"""
        return self._dr

    @property
    def psfSigma(self) -> float:
        """The sigma value (standard deviation) [pixels] of the camera's 2D Gaussian PSF"""
        return self._psfSigma

    @psfSigma.setter
    def psfSigma(self, new: float):
        """Sets a new value for the camera's 2D Gaussian PSF's standard deviation [pixels]"""
        self._psfSigma = new

    @property
    def tr(self) -> float:
        """The total optical transmission of the camera."""
        return self._tr

    @tr.setter
    def tr(self, value: float):
        """Updates the value of the camera's total optical transmission."""
        self._tr = value

    @property
    def nr(self) -> _fnp:
        """Camera's RMS read noise [e-]. This can be either a single value (applying to all pixels) or a numpy
        array with shape (dr, dc) providing a value for each pixel of the detector."""
        return self._nr

    @nr.setter
    def nr(self, value: _fnp):
        """Updates the camera's RMS read noise [e-]. This can be either a single value (applying to all pixels) or a
        numpy array with shape (dr, dc) providing a value for each pixel of the detector."""
        self._nr = value

    @property
    def jd(self) -> _fnp:
        """Camera's detector dark current [e- s^-1]. This can be either a single value (applying to all pixels) or a
        numpy array with shape (nr, nc) providing a value for each pixel of the detector."""
        return self._jd

    @jd.setter
    def jd(self, value: _fnp):
        """Updates the camera's dark current [e- s^-1]. This can be either a single value (applying to all pixels) or a
        numpy array with shape (nr, nc) providing a value for each pixel of the detector."""
        self._jd = value

    @property
    def binning(self) -> int:
        """pixel binning factor (=1 means no binning). Binning is performed on-chip prior to readout"""
        return self._binning

    @binning.setter
    def binning(self, value: int):
        """Updates the camera's pixel binning factor (=1 means no binning)."""
        self._binning = value

    @property
    def qe(self) -> float:
        """detector quantum efficiency (value from 0->1)"""
        return self._qe

    @qe.setter
    def qe(self, value: float):
        """sets new value for detector's quantum efficiency (value from 0->1)"""
        self._qe = value

    @property
    def bitdepth(self) -> int:
        """bits per pixel in digital image"""
        return self._bitdepth

    @bitdepth.setter
    def bitdepth(self, value: int):
        """sets new value for camera's bit depth"""
        self._bitdepth = value

    @property
    def fwc(self):
        """full well capacity of each pixel [e-]"""
        return self._fwc

    @fwc.setter
    def fwc(self, value: int):
        """sets new value for camera's full well capacity [e-]"""
        self._fwc = value

    @property
    def gain(self):
        """The gain of the camera in electrons per digital unit"""
        return self.fwc / 2 ** self.bitdepth

    def worldToImage(self, point: 'Vec3', cull=True) -> Tuple[Optional[_fnp], Optional[_fnp]]:
        """ Returns the camera's image coordinate that views a given world point

        :param point: the world point [m]
        :param cull: whether to cull (return np.nan/None) the image coordinates if they are outside the
                    boundary of the camera's detector
        :return: the image coordinates (colum, row) viewing the world point [pixels]
        """
        message = "worldToImage() not implemented - use a subclass of Camera with implemented worldToImage function"
        warnings.warn(message)
        return 0.5 * self._dc, 0.5 * self._dr

    def _getScaleFactorToDetector(self, arr: np.ndarray) -> tuple[float, float]:
        """ Returns the factor by which the given array's shape is scaled relative to the shape of the camera's
        detector. Returns as a tuple of (columns scale factor, rows scale factor).

         E.g. if the camera's detector is of shape (100, 100) and the given array is of shape (200, 300),
         this function will return (3, 2).

        :param arr: The array for which scale factor should be found.
        :return: Scale factor tuple.
        """
        sf_r = arr.shape[0] / self.dr
        sf_c = arr.shape[1] / self.dc
        return sf_c, sf_r

    def _assertArrayMatchesDetector(self, arr: np.ndarray, arr_name: str):
        message = ("Shape of {0} must be an integer multiple of this camera's detector array's shape. "
                   "Detector shape: {1}; {0} shape: {2}").format(arr_name, (self._dr, self._dc), arr.shape)
        sf_r = arr.shape[0] / self.dr
        sf_c = arr.shape[1] / self.dc
        cond = sf_r.is_integer() and sf_c.is_integer()
        assert cond, message

    def downsampleToDetectorPixels(self, ssi: np.ndarray, average=False):
        """ Takes a super-sampled image (more than one sample per pixel of this camera) and downsamples it to a single
        sample per camera pixel (by summing subpixel values to produce a single value for each camera pixel).
        The super-sampled image's shape must be an integer multiple of this camera's detector's
        shape (for example, for a camera with 1000 x 1000 pixel detector, the super-sampled image must have shape
        n1000 x n1000, where n is an integer, and the returned downsampled image will be of shape 1000 x 1000).

        :param ssi: Super-sampled image. Its shape must be integer multiple of this camera's detector's shape.
        :param average: If true, each pixel's final value will be the average of the super pixels it comprised, otherwise
            it will be the sum.
        :return: The downsampled image, with one value corresponding to each pixel of this camera.
        """
        sf_c, sf_r = self._getScaleFactorToDetector(ssi)
        self._assertArrayMatchesDetector(ssi, "super-sampled image")
        sf = int(sf_r)
        if sf == 1:
            return ssi
        dsi = math2.binNumpyArray2D(ssi, sf)
        if average:
            dsi = dsi / sf ** 2
        return dsi

    def applyPSF(self, image: np.ndarray) -> np.ndarray:
        """ Applies the effect of a gaussian point spread function to the given image. The image should have the same
        shape as this camera's detector, or its shape should be an integer multiple of this camera's detector's shape
        (for multiple samples per camera pixel).

        The PSF is approximated as a 2D gaussian with sigma value equal to the camera's psfSigma property (update this
        value first as required). The sigma value has units of camera detector pixels.

        :param image: the original image
        :return: the image with PSF applied
        """
        if self._psfSigma == 0:
            return image
        self._assertArrayMatchesDetector(image, "flux image")
        assert self._psfSigma >= 0, "psf radius cannot be negative"
        sf, _ = self._getScaleFactorToDetector(image)

        if self._psfSigma != 0:
            image = gaussian_filter(image, sigma=self._psfSigma * sf)  # approximate effect of PSF as 2D gaussian
        return image

    def propagateFluxToDetector(self, flux: np.ndarray):
        """ Takes an at-aperture flux image (i.e. an image of the fluxes [W m^-2] seen by each of this camera's
        pixels/sub-pixels before the flux has entered the camera) and propagates it to the camera's detector, accounting
        for: PSF, aperture, optical transmission. The returned array is the power [W] incident on each pixel of the
        detector.

        Update camera's psfSigma, epa/epd and tr properties as required before calling.

        :param flux: flux [W m^-2] of radiant energy travelling toward each of this camera's pixels (or sub-pixels if a
            super-sampled flux image is provided) just before entering the camera's optics. This array must have same
            shape as detector, or an integer multiple of this shape (for super-sampled flux image).
        :return: array of power [W] incident on each pixel of the camera's detector.
        """
        flux = self.applyPSF(flux)
        flux = self.downsampleToDetectorPixels(flux)
        power = self.tr * flux * self.epa  # W per pixel
        return power

    def countElectrons(self, photon_rate: np.ndarray, t_exp: float, b: int = 1):
        """ For the given pixel photon rates [photon s^-1] and exposure time, this function returns the number of
        electrons counted by the pixels (including noise sources; electron count in each pixel is limited to camera's
        full well capacity).

        Ensure that the camera's quantum efficiency, dark current, read noise and full well capacity properties have
        been updated as required before calling.

        :param photon_rate: rate at which photons are incident on pixels [photon s^-1]
        :param t_exp: image exposure time [s]
        :param b: integer on-chip binning factor, performed prior to readout (=1 means no binning)
        :return: no. of electrons counted by pixels
        """
        photon_rate[photon_rate < 0] = 0
        notNan = ~np.isnan(photon_rate)
        poissonValLim = 1e10
        overLim = (photon_rate * t_exp) > poissonValLim
        canPoisson = notNan * ~overLim

        rng = np.random.default_rng()
        pDiscrete = np.zeros_like(photon_rate, dtype=float)
        pDiscrete[canPoisson] = rng.poisson((photon_rate[canPoisson] * t_exp))
        pDiscrete[overLim] = poissonValLim

        eSignal = self.qe * pDiscrete
        eThermSignal = rng.poisson(t_exp * self.jd, size=photon_rate.shape)

        if b != 1:
            eSignal = math2.binNumpyArray2D(eSignal, b)
            eThermSignal = math2.binNumpyArray2D(eThermSignal, b)
        eReadNoise = np.round(rng.normal(0, self.nr, size=eSignal.shape))
        eCount = eSignal + eThermSignal + eReadNoise
        eCount[eCount > self.fwc] = self.fwc

        return eCount

    def digitise(self, electron_count: np.ndarray):
        """ Converts the given pixel electron counts to digital values (ADU).

        This camera's full well capacity (fwc) and bit depth properties should first be updated as required.

        :param electron_count: The number of electrons counted by each pixel.
        :return: The digitised measurement (ADU) for each pixel.
        """
        maxADU = 2 ** self.bitdepth - 1
        return np.floor(maxADU * electron_count / self.fwc)

    def image(self, flux: np.ndarray, t_exp: float, ew: float):
        """ Takes an at-aperture flux image (i.e. an image of the fluxes [W m^-2] seen by each of this camera's
        pixels/sub-pixels before the flux has entered the camera) and generates the resulting digital image that the
        camera would capture.

        Ensure that camera entrance pupil, optical transmission, PSF, dark current, read noise, quantum efficiency,
        full well capacity and bit depth have first been set as necessary.

        :param flux: flux [W m^-2] of radiant energy travelling toward each of this camera's pixels (or sub-pixels if a
            super-sampled flux image is provided) just before entering the camera's optics. This array must have same
            shape as detector, or an integer multiple of this shape (for super-sampled flux image). This flux should be
            the total flux over the spectral band being used for imaging, approximated as having a single wavelength
            equal to the given effective wavelength.
        :param t_exp: image exposure time [s]
        :param ew: effective wavelength [nm] of the light being imaged (i.e. the flux is treated as monochromatic with
            wavelength = ew)
        :return: the resulting image
        """
        power = self.propagateFluxToDetector(flux)
        pRate = power / rd.photonEnergy(ew)
        eCount = self.countElectrons(pRate, t_exp, b=self.binning)
        return self.digitise(eCount)


class CameraTwoWay(Camera):
    """Base class for a general projective camera for which both inward and outward projections are defined.

    Do not use this class directly - instead use subclasses of it."""

    def __init__(self, dc: int, dr: int):
        super().__init__(dc, dr)

    def worldFromImage(self, column: _fnp, row: _fnp) -> 'Ray':
        """ Returns the ray in space along which a given image point is viewing

        :param column: the column coordinate of the image point [pixels]
        :param row: the row coordinate of the image point [pixels]
        :return: the view ray (in world coordinates)
        """
        message = ("worldFromImage() not implemented - use a subclass of CameraTwoWay with implemented worldFromImage"
                   " function")
        warnings.warn(message)
        return Ray(self.pos, self.frame.w)

    def pixelsLOS(self, sf=1, region: Tuple[int, int, int, int] = None):
        """Returns a numpy-type ray (see Ray.isNumpyType documentation) describing the lines of sight (LOSs) of each of
        the camera's pixels within the detector
        (or sub-region of the detector defined by the given region tuple if not None).

        :param sf: sampling factor, controlling the number of rays (or lines of sight) returned per pixel (where number
            of rays per pixel = sf^2). This can be used to render images with subpixel sampling for higher fidelity.
            sf=1 by default.
        :param region: optional tuple describing rectangular sub-region of detector for which LOSs will be returned.
        :return: A numpy-type ray containing all the calculated pixel LOSs from the camera (with its vectors' components
            represented by numpy arrays of shape (no. of region rows * sf, no. of region columns * sf).
        """
        minCol, maxCol, minRow, maxRow = 0, self._dc, 0, self._dr
        if region is not None:
            minCol, maxCol, minRow, maxRow = region
        nc = maxCol - minCol
        nr = maxRow - minRow
        delta = 0.5 / sf
        c = np.linspace(minCol + delta, maxCol - delta, int(nc * sf))
        r = np.linspace(minRow + delta, maxRow - delta, int(nr * sf))
        c, r = np.meshgrid(c, r)

        return self.worldFromImage(c, r)

    def calculateIFOV(self, sf=1):
        """ Calculates the IFOV of each of this camera's pixels (or subpixels if sf>1) over the camera's whole detector
        and returns as a 2D numpy array.

        :param sf: sampling factor controlling the number of subpixels (if any). Number of subpixels per actual camera
            pixel = sf^2. sf=1 (no subpixel sampling) by default.
        :return: tuple containing two 2D numpy arrays giving the horizontal and vertical IFOVs of the camera's pixels
            (or subpixels) [radians]
        """
        c = np.arange(sf * self.dc + 1) / sf
        r = np.arange(sf * self.dr + 1) / sf
        c, r = np.meshgrid(c, r)
        d = self.worldFromImage(c, r).d
        v1 = Vec3((d.x[:-1, :-1], d.y[:-1, :-1], d.z[:-1, :-1]))
        v2 = Vec3((d.x[:-1, 1:], d.y[:-1, 1:], d.z[:-1, 1:]))
        v3 = Vec3((d.x[1:, :-1], d.y[1:, :-1], d.z[1:, :-1]))
        a1 = v1.angleWith(v2)
        a2 = v1.angleWith(v3)
        return a1, a2

    def convertRadianceImageToEquivalentFlux(self, radiance: np.ndarray):
        """ Takes a radiance image of a scene as seen by this camera (i.e. an image containing the radiance
        [W m^-2 sr^-1] of the surface observed by each of this camera's pixels) and converts it to an equivalent flux
        [W m^-2] seen by each pixel (or subpixel if a super-sampled image is provided).

        The radiance image provided should be the same shape as this camera's detector (i.e. one radiance value per
        camera pixel), or an integer multiple of this camera's detector's shape (multiple values per pixel, each
        representing a subpixel).

        :param radiance: radiance image [W m^-2 sr^-1]
        :return: equivalent flux image [W m^-2]
        """
        self._assertArrayMatchesDetector(radiance, "radiance image")
        sf = self._getScaleFactorToDetector(radiance)[0]
        sf = int(sf)
        ifov_h, ifov_v = self.calculateIFOV(sf=sf)
        solidAng = ifov_h * ifov_v  # sr
        return radiance * solidAng  # W m^-2

    def convertFluxImageToEquivalentRadiance(self, flux: np.ndarray):
        """ Takes a flux image of a scene as seen by this camera (i.e. an image containing the flux
        [W m^-2] of the light observed by each of this camera's pixels) and converts it to an equivalent radiance
        [W m^-2 sr^-1] seen by each pixel (or subpixel if a super-sampled image is provided).

        :param flux: flux image [W m^-2]
        :return: equivalent radiance image [W m^-2 sr^-1]
        """
        self._assertArrayMatchesDetector(flux, "flux image")
        sf = self._getScaleFactorToDetector(flux)[0]
        sf = int(sf)
        ifov_h, ifov_v = self.calculateIFOV(sf=sf)
        solidAng = ifov_h * ifov_v  # sr
        return flux / solidAng  # W m^-2 sr^-1

    def viewOf(self, camera: 'Camera', sf=1) -> Tuple[np.ndarray, np.ndarray]:
        """ Provides this camera's view of another camera's FOV (assuming both cameras share the same optical centre),
        by returning the columns and rows of the other camera's image that this camera's pixels view.

        :param camera: the other camera whose image this camera is viewing.
        :param sf: sampling factor controlling the number of values calculated per pixel of this camera (where number of
            values per pixel = sf^2). This can be used to perform subpixel sampling for higher fidelity. sf=1 by
            default.
        :return: tuple of numpy arrays containing the column and row of the other camera's image that each pixel of this
            camera (or subpixel) views.
        """
        home = camera.pos
        camera.pos = self.pos
        point = self.pixelsLOS(sf=sf).point(1)
        col, row = camera.worldToImage(point, cull=True)
        camera.pos = home
        return col, row

    def drawFOV(self, camera: 'Camera', sf=1) -> np.ndarray:
        """ Returns an image of the given camera's detector with this camera's FOV drawn onto it.

        :param camera: Camera whose detector the FOV will be drawn on to.
        :param sf: sampling factor controlling the number of pixels in the returned image relative to the number of
            pixels in the given camera's detector. If the given camera's detector has c columns and r rows of pixels,
            the image returned by this function will have sf x c columns and sf x r rows. Can be used to increase
            sampling rate to improve image fidelity. sf=1 by default.
        :return: FOV image.
        """
        home = camera.pos
        camera.pos = self.pos
        img = np.zeros((camera.dr, camera.dc), dtype=int)
        p = self.pixelsLOS(sf=sf).point(1)
        colout = np.linspace(0, self.dc - 1, self.dc * sf)
        rowout = np.linspace(0, self.dr - 1, self.dr * sf)
        colout, rowout = np.meshgrid(colout, rowout)
        condition = (colout < 50) + (colout > self.dc - 50) + (rowout < 50) + (rowout > self.dr - 50)
        colin, rowin = camera.worldToImage(p.npMask(condition))
        camera.pos = home
        img[rowin.astype(int), colin.astype(int)] = 1
        return img


class CameraSimple(CameraTwoWay):
    """Base class for simple camera models which can be described by a focal length and a rectangular detector on a
    focal plane, with no lens distortions (e.g. perfect pinhole and perfect fisheye geometries).

    Do not use this base class directly - use the Pinhole or Fisheye subclasses.
    """

    def __init__(self, f: float, dc: int, dr: int, dwx: float, dwy: float):
        self._f = f  # the camera's focal length [m]
        self._dwx = dwx  # detector width (physical length of a row of pixels) [m]
        self._dwy = dwy  # detector height (physical length of a column of pixels) [m]
        self._ppo = 0.5 * Vec2((dwx, dwy))  # the principal point's position in detector coordinate frame [m]
        super().__init__(dc, dr)

    @property
    def dwx(self):
        """The physical width of the camera's detector [m]"""
        return self._dwx

    @property
    def dwy(self):
        """The physical height of the camera's detector [m]"""
        return self._dwy

    @property
    def pwx(self):
        """The physical width of a pixel [m]"""
        return self._dwx / self.dc

    @property
    def pwy(self):
        """The physical height of a pixel [m]"""
        return self._dwy / self.dr

    @property
    def f(self):
        """The focal length of the camera [m]"""
        return self._f

    @property
    def f_pixels(self):
        # the camera's focal length in units of pixels [pixels]
        return self.f * self.dc / self.dwx

    @property
    def ppo(self) -> 'Vec2':
        """ The principal point offset (principal point's position in detector coordinate frame) [m].
        The principal point is the point where the optical axis intersects the focal plane.

        :return: The principal point's position in detector coordinate frame [m]
        """
        return self._ppo

    @ppo.setter
    def ppo(self, value: 'Vec2'):
        """ Sets a new principle point offset

        :param value: The new offset
        """
        self._ppo = value

    @property
    def fov(self) -> Tuple[float, float]:
        """The camera's horizontal and vertical fields of view [radians] (to be overriden by subclasses)"""
        message = "fov property not implemented - use a subclass of CameraSimple with implemented fov property"
        warnings.warn(message)
        return 0, 0

    @property
    def ifov(self) -> Tuple[float, float]:
        """The approximate horizontal and vertical ifov of a pixel at centre of FOV [radians].

        Use calculateIFOV() for more accurate, pixel-specific IFOVs.
        """
        fovx, fovy = self.fov
        return fovx / self._dc, fovy / self._dr

    def projectIn(self, point: 'Vec3') -> 'Vec2':
        """ Projects a 3d point in world coordinates to the camera's detector coordinate frame (to be overridden by
        subclasses)

        :param point: the 3d point in world coordinates
        :return: the projected coordinates in the detector coordinate frame
        """
        message = "projectIn() not implemented - use a subclass of CameraSimple with implemented projectIn function"
        warnings.warn(message)
        return self._ppo

    def projectOut(self, point: 'Vec2') -> 'Ray':
        """ Projects a 2d point in the camera's detector coordinate frame to a ray in 3d space in world coordinates (to
        be overridden by subclasses)

        :param point: the point in the detector coordinate frame
        :return: the corresponding ray in 3d space in world coordinates
        """
        message = "projectOut() not implemented - use a subclass of CameraSimple with implemented projectOut function"
        warnings.warn(message)
        return Ray(self.pos, self.frame.w)

    def convertDetectorCoordPhysicalToImage(self, point: 'Vec2', cull=False) -> \
            Tuple[Optional[_fnp], Optional[_fnp]]:
        """ Converts a point in the detector coordinate frame from physical units (m) to image units (pixels)

        :param point: the physical coordinate to convert [m]
        :param cull: whether to cull (return np.nan/None) the converted coordinates if they are outside the
                    boundary of the detector
        :return: the point in image coordinates (column, row)
        """
        c = self._dc * point.x / self._dwx
        r = self._dr * point.y / self._dwy

        if cull:
            if type(c) is np.ndarray or type(r) is np.ndarray:
                out = (c < 0) + (c >= self._dc) + (r < 0) + (r >= self._dr)
                c[out] = np.nan
                r[out] = np.nan
            else:
                if c < 0 or c >= self._dc or r < 0 or r >= self._dr:
                    return None, None
        return c, r

    def convertDetectorCoordImageToPhysical(self, column: _fnp, row: _fnp) -> 'Vec2':
        """ Converts a point in the detector coordinate frame from image units (pixels) to physical units (m)

        :param column: the column of the coordinate point
        :param row: the row of the coordinate point
        :return: the coordinate in physical units (m)
        """
        x = self._dwx * column / self._dc
        y = self._dwy * row / self._dr
        return Vec2((x, y))

    def worldToImage(self, point: 'Vec3', cull=True) -> Tuple[_fnp, _fnp]:
        """ Returns the image coordinate that views a given world point

        :param point: the world point [m]
        :param cull: whether to cull (return np.nan/None) the converted coordinates if they are outside the
                    boundary of the detector
        :return: the image coordinates (column, row) viewing the world point
        """
        return self.convertDetectorCoordPhysicalToImage(self.projectIn(point), cull=cull)

    def worldFromImage(self, column: _fnp, row: _fnp) -> 'Ray':
        """ Returns the ray in space along which a given image point is viewing

        :param column: the column of the image point
        :param row: the row of the image point
        :return: the view ray
        """
        return self.projectOut(self.convertDetectorCoordImageToPhysical(column, row))


class Pinhole(CameraSimple):
    """Class for a camera with an ideal pinhole imaging geometry."""

    def __init__(self, f: float, dc: int, dr: int, dwx: float, dwy: float):
        """Initialises a new pinhole camera with given parameters

        :param f: camera focal length [m]
        :param dc: number of columns of pixels on detector
        :param dr: number of rows of pixels on detector
        :param dwx: width of detector (physical length of a row of pixels) [m]
        :param dwy: height of detector (physical length of a column of pixels) [m]
        """
        super().__init__(f, dc, dr, dwx, dwy)
        self._fa = 1  # focal aspect
        self._skew = 0

    @property
    def intrinsic(self) -> Mat3:
        """ The camera's intrinsic matrix (see e.g. https://en.wikipedia.org/wiki/Camera_resectioning and
        https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/) """
        row1 = self._f, self._skew, self._ppo.x
        row2 = 0, self._f * self._fa, self._ppo.y
        row3 = 0, 0, 1
        return Mat3(row1, row2, row3)

    @property
    def intrinsicInverse(self) -> Mat3:
        """The inverse of the camera's intrinsic matrix (see e.g.
        https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/)

        :return: the inverse intrinsic matrix
        """
        row1 = self._f * self._fa, -self._skew, (self._ppo.y * self._skew) - (self._ppo.x * self._f * self._fa)
        row2 = 0, self._f, -self._ppo.y * self._f
        row3 = 0, 0, self._f * self._f * self._fa
        c = 1 / (row3[2])
        return c * Mat3(row1, row2, row3)

    @property
    def focalAspect(self):
        """The ratio of vertical focal length to horizontal focal length (only relevant for a camera whose focal length
         is different in the horizontal detector axis to the vertical detector axis, see e.g.
         https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/)"""
        return self._fa

    @focalAspect.setter
    def focalAspect(self, value: float):
        """ Sets a new focalAspect (the ratio of vertical focal length to horizontal focal length)

        :param value: new focal aspect
        """
        self._fa = value

    @property
    def fov(self) -> Tuple[float, float]:
        """The camera's horizontal and vertical fields of view [radians]"""
        fovx = 2 * math.atan(0.5 * self.dwx / self._f)
        fovy = 2 * math.atan(0.5 * self.dwy / self._f / self._fa)
        return fovx, fovy

    def projectIn(self, point: 'Vec3') -> 'Vec2':
        """ Projects a 3d point in world coordinates to the camera's detector coordinate frame

        :param point: the 3d point in world coordinates
        :return: the projected coordinates in the detector coordinate frame
        """
        local = self._frame.fromWorld(point)  # convert point from world to camera frame
        if local.isNumpyType:
            behind = local.z < 0
            local.x[behind] = np.nan
            local.y[behind] = np.nan
            local.z[behind] = np.nan
        elif local.z < 0:
            return Vec2((np.nan, np.nan))
        projected = self.intrinsic * local  # project point to homogeneous 2D coords using intrinsic camera matrix
        projected = projected / projected.z  # convert homogeneous coords to inhomogeneous (actual) coords on detector
        return Vec2(projected.tuple[:2])

    def projectOut(self, point: 'Vec2') -> 'Ray':
        """ Projects a 2d point in the camera's detector coordinate frame to a ray in 3d space in world coordinates.

        :param point: the point in the detector coordinate frame
        :return: the corresponding ray in 3d space in world coordinates
        """
        pointhom = Vec3(point.tuple + (1,))  # convert inhomogeneous detector coords to homogeneous coords
        d = self.intrinsicInverse * pointhom  # project homogeneous detector coords to point in space
        local = Ray(Vec3.zero(), d.norm)  # ray to point in space from camera principal point in camera frame
        return local.transformed(self._frame, Frame.world())  # ray in world frame


class Fisheye(CameraSimple):
    """Class for a camera with an ideal equidistant fisheye imaging geometry."""

    @property
    def fov(self) -> Tuple[float, float]:
        """The camera's horizontal and vertical fields of view [radians] (assuming equidistant fisheye geometry)"""
        fovx = self._dwx / self._f
        fovy = self._dwy / self._f
        return fovx, fovy

    def projectIn(self, point: 'Vec3') -> 'Vec2':
        """ Projects a 3d point in world coordinates to the camera's detector coordinate frame

        :param point: the 3d point in world coordinates
        :return: the projected coordinates in the detector coordinate frame
        """
        local = self._frame.fromWorld(point)  # convert point from world to camera frame
        alpha = np.arctan2((local.x ** 2 + local.y ** 2) ** 0.5, local.z)  # angle of incidence of the observed point
        dirvec = Vec2(local.tuple[:2])  # direction from principal point to observed point's projection on focal plane
        onax = (dirvec.x == 0) * (dirvec.y == 0)  # if the observed point is on the optical axis, dirvec's length is 0
        # normalise dirvec to length 1 and handle on-axis case:
        if dirvec.isNumpyType:
            dirvec = dirvec.norm
            dirvec.x[onax] = 0
            dirvec.y[onax] = 0
        else:
            if onax:
                dirvec = Vec2((0, 0))
            else:
                dirvec = dirvec.norm
        rim = alpha * self._f  # radius of observed point's projection on focal plane from principal point
        uv = dirvec * rim  # coordinate of point's projection on detector plane in camera coordinate frame
        return uv + self._ppo  # coordinate of point's projection in detector coordinate frame

    def projectOut(self, point: 'Vec2') -> Optional['Ray']:
        """ Projects a 2d point in the camera's detector coordinate frame to a ray in 3d space in world coordinates.

        :param point: the point in the detector coordinate frame
        :return: the corresponding ray in 3d space in world coordinates
        """
        u, v = (point - self._ppo).tuple
        alpha = ((u ** 2 + v ** 2) ** 0.5) / self._f
        rotax = Vec3.k().cross(Vec3((u, v, 0)))
        local = Ray(Vec3.zero(), Vec3.k().rotated(rotax, alpha))
        if local.isNumpyType:
            behind = local.d.z < 0
            local.d.x[behind] = np.nan
            local.d.y[behind] = np.nan
            local.d.z[behind] = np.nan
        else:
            if local.d.z < 0:
                return None
        return local.transformed(self._frame, Frame.world())  # ray in world frame


class PinholeOpenCV(CameraTwoWay):
    """Class for a camera described by the OpenCV pinhole camera model."""
    def __init__(self, matrix: np.ndarray, distortions: np.ndarray, dc: int, dr: int):
        """ Initialises a new camera with the opencv pinhole model

        :param matrix: camera matrix (3x3 numpy array)
        :param distortions: distortion coefficients
        :param dc: number of detector columns
        :param dr: number of detector rows
        """
        super().__init__(dc, dr)
        self._fx = matrix[0, 0]
        self._fy = matrix[1, 1]
        self._cx = matrix[0, 2]
        self._cy = matrix[1, 2]
        self._alpha = matrix[0, 1]
        self._distortions = distortions

    @property
    def matrix(self) -> Mat3:
        """The camera matrix as a Mat3 object"""
        return Mat3((self._fx, self._alpha, self._cx), (0, self._fy, self._cy), (0, 0, 1))

    @property
    def matrixAsNumpy(self) -> np.ndarray:
        """The camera matrix as a 3x3 numpy array"""
        return np.array(((self._fx, self._alpha, self._cx), (0, self._fy, self._cy), (0, 0, 1)))

    @property
    def matrixInverse(self) -> Mat3:
        """The inverse of the camera matrix as a Mat3 object"""
        row1 = self._fy, -self._alpha, (self._cy * self._alpha) - (self._cx * self._fy)
        row2 = 0, self._fx, -self._cy * self._fx
        row3 = 0, 0, self._fx * self._fy

        c = 1 / (row3[2])
        return c * Mat3(row1, row2, row3)

    @property
    def matrixInverseAsNumpy(self) -> np.ndarray:
        """The inverse of the camera matrix as a 3x3 numpy array"""
        return self.matrixInverse.asNumpy

    def worldToImage(self, point: 'Vec3', cull=False) -> Tuple[Optional[_fnp], Optional[_fnp]]:
        local = self._frame.fromWorld(point)
        if point.isNumpyType:
            coords = np.empty((point.x.size, 1, 3), dtype='float32')
            coords[:, 0, 0] = local.x.flatten()
            coords[:, 0, 1] = local.y.flatten()
            coords[:, 0, 2] = local.z.flatten()
        else:
            coords = np.array([[[local.x, local.y, local.z]]], dtype='float32')
        rvec = np.array((0, 0, 0), dtype='float32')
        tvec = np.array((0, 0, 0), dtype='float32')
        imCoords, _ = cv2.projectPoints(coords, rvec, tvec, self.matrixAsNumpy, self._distortions)
        if point.isNumpyType:
            u = imCoords[:, 0, 0].reshape(point.numpyShape)
            v = imCoords[:, 0, 1].reshape(point.numpyShape)
            if cull:
                toCull = (u < 0) + (u >= self.dc) + (v < 0) + (v >= self.dr)
                u[toCull] = np.nan
                v[toCull] = np.nan
        else:
            u = imCoords[0, 0, 0]
            v = imCoords[0, 0, 1]
            if cull:
                if u < 0 or u >= self.dc or v < 0 or v >= self.dr:
                    return None, None
        return u, v

    def worldFromImage(self, column: _fnp, row: _fnp) -> 'Ray':
        # cv2.undistortPoints has to receive points as a 1D list, so if column and row are 2D or more, flatten them:
        origArrShape = None
        if type(column) is np.ndarray or type(row) is np.ndarray:
            if type(column) is not np.ndarray:
                column = np.full_like(row, column)
            elif type(row) is not np.ndarray:
                row = np.full_like(column, row)
            if column.ndim > 1:
                origArrShape = column.shape
                column = column.flatten()
                row = row.flatten()

        points = np.array((column, row), dtype='float32')  # Nx2 array for passing to cv2.undistortPoints

        undist = cv2.undistortPoints(points, self.matrixAsNumpy, self._distortions, P=self.matrixAsNumpy)
        col_undist, row_undist = undist[..., 0, 0], undist[..., 0, 1]  # undistorted columns and rows of image points

        if col_undist.shape == (1,):  # if there is only a single column value, convert it from numpy array to float
            col_undist = col_undist[0]
        if row_undist.shape == (1,):  # if there is only a single row value, convert it from numpy array to float
            row_undist = row_undist[0]

        # if column and row arrays had to be flattened for cv2.undistortPoints, return to original shape:
        if origArrShape is not None:
            col_undist = np.reshape(col_undist, origArrShape)
            row_undist = np.reshape(row_undist, origArrShape)

        homogeneous = Vec3((col_undist, row_undist, 1))  # convert inhomogeneous detector coords to homogeneous coords
        d = self.matrixInverse * homogeneous  # project homogeneous detector coords to point in space
        local = Ray(Vec3.zero(), d.norm)  # ray to point in space from camera principal point in camera frame
        return local.transformed(self._frame, Frame.world())  # ray in world frame


class FisheyeOpenCV(Camera):
    """Class for a camera described by the OpenCV fisheye camera model
        https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
    """

    def __init__(self, matrix: np.ndarray, distortions: np.ndarray, dc: int, dr: int):
        """ Initialises a new camera with the opencv fisheye camera model

        :param matrix: camera matrix (3x3 numpy array)
        :param distortions: camera distortion coefficients
        :param dc: number of detector columns
        :param dr: number of detector rows
        """
        super().__init__(dc, dr)
        self._fx = matrix[0, 0]
        self._fy = matrix[1, 1]
        self._cx = matrix[0, 2]
        self._cy = matrix[1, 2]
        self._alpha = matrix[0, 1]
        self._k1, self._k2, self._k3, self._k4 = distortions[:, 0]

    @property
    def matrix(self):
        """The camera matrix as a Mat3 object"""
        return Mat3((self._fx, self._alpha, self._cx), (0, self._fy, self._cy), (0, 0, 1))

    @property
    def matrixAsNumpy(self):
        """The camera matrix as a 3x3 numpy array"""
        return np.array(((self._fx, self._alpha, self._cx), (0, self._fy, self._cy), (0, 0, 1)))

    @property
    def distortions(self):
        """A tuple of the camera's 4 openCV fisheye distortion parameters"""
        return self._k1, self._k2, self._k3, self._k4

    def worldToImage(self, point: 'Vec3', cull=False) -> Tuple[Optional[_fnp], Optional[_fnp]]:
        local = self._frame.fromWorld(point)
        coords = np.empty((point.y.size, 1, 3), dtype='float32')
        coords[:, 0, 0] = local.x.flatten()
        coords[:, 0, 1] = local.y.flatten()
        coords[:, 0, 2] = local.z.flatten()
        rvec = np.array((0, 0, 0), dtype='float32')
        tvec = np.array((0, 0, 0), dtype='float32')
        distortions = np.empty((1, 4))
        distortions[0, :] = self.distortions
        imCoords, _ = cv2.fisheye.projectPoints(coords, rvec, tvec, self.matrixAsNumpy, distortions)
        if point.isNumpyType:
            u = imCoords[:, 0, 0].reshape(point.numpyShape)
            v = imCoords[:, 0, 1].reshape(point.numpyShape)
            if cull:
                toCull = (u < 0) + (u >= self.dc) + (v < 0) + (v >= self.dr)
                u[toCull] = np.nan
                v[toCull] = np.nan
        else:
            u = imCoords[0, 0, 0]
            v = imCoords[0, 0, 1]
            if cull:
                if u < 0 or u >= self.dc or v < 0 or v >= self.dr:
                    return None, None
        return u, v
