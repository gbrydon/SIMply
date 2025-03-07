# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module containing classes for using image-based textures of surfaces."""

import numpy as np
from coremaths.vector import Vec3
from coremaths.frame import Frame
import cv2
import planetary_data as planData
from typing import Callable, Optional, Tuple, Union

_fnp = Union[float, np.ndarray]
_inp = Union[int, np.ndarray]
_lims = Tuple[float, float, float, float]


class Texture:
    """Class for representing image-based textures of surfaces."""
    def __init__(self, image: Union[str, np.ndarray]):
        """ Initialises a new texture described by the given image.

        :param image: the image (either a 2D numpy array, or a path to the image file)
        """
        if type(image) is np.ndarray:
            self._imageData = image
            self._image_path = None
        else:
            self._image_path = image
            self._imageData = None

        def valueModifier(v: _fnp) -> _fnp:
            return v
        self._modifier = valueModifier

    @staticmethod
    def planetocentric(image: Union[str, np.ndarray], extent_lims: _lims, frame=Frame.world()):
        """ Returns a planetocentric texture, for use with a planetocentric surface.

        A planetocentric texture covers a region on a planetary surface defined by minimum and maximum longitude and
        latitude bounds, where longitude and latitude are measured relative to a coordinate frame at the planet's
        centre. Longitude is measured from the frame's x-axis, latitude is measured from the frame's x-y plane.

        :param image: the image (either a 2D numpy array, or a path to the image file).
        :param extent_lims: (longwest, longeast, minlat, maxlat) tuple of texture's extent [degrees].
        :param frame: the coordinate frame at the planet's centre.
        :return: the texture.
        """
        return TexturePlanetocentric(image, extent_lims, frame=frame)

    @staticmethod
    def planetocentricFromMetadata(path: str):
        """ Returns a Planetocentric texture initialised from a metadata text file at the given path.

        :param path: path to metadata text file.
        :return: the texture
        """
        return TexturePlanetocentric.fromMetadataTextFile(path)

    def setValueModifier(self, modifier: Callable[[_fnp], _fnp]):
        """ Sets this texture's modifier function, which takes as input the texture's source image values and calculates
        modified values according to the modifier function given here.

        An example use case of this is for a planetary ORI whose image values are to be used to calculate a physical
        unit (such as I/F) by using an offset and scaling factor defined in the ORI's metadata.

        :param modifier: function that modifies the values of this texture, by taking in the raw texture value(s) as a
            single float or numpy array and returning a new value(s), also as a single float or numpy array.
        """
        self._modifier = modifier

    def imageData(self) -> np.ndarray:
        """Returns this texture's image as a numpy array"""
        if self._imageData is not None:
            return self._imageData
        image = cv2.imread(self._image_path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def value(self, column: _inp, row: _inp, raw=False, rgb_channel=2) -> _fnp:
        """The texture's value at the given column and row.

        If raw argument is false, the returned value will be modified according to this texture's value modifier
        function (if any). If raw argument is true, the unmodified value of the texture's source image will be returned.

        :param column: the column coordinate on the texture image
        :param row: the row coordinate on the texture image
        :param raw: whether to return the raw value of the source image (otherwise value will be modified by texture's
            value modifier function (if any))
        :param rgb_channel: If the texture is an rgb texture (i.e. has 3 channels), rgb_channel dictates which channel
            the texture value will be taken from. Set rgb_channel to 1, 2 or 3 to use the texture's red, green or blue
            channel respectively. If the texture is not rgb (i.e. has a single channel), the value of rgb_channel is
            ignored.
        :return: the value
        """
        array = self.imageData()
        if array.ndim == 3:
            assert 1 <= rgb_channel <= 3, "rgb_channel must be equal to 1, 2 or 3"
            array = array[:, :, rgb_channel - 1]
        rawValue = array[row, column]
        if raw:
            return rawValue
        return self._modifier(rawValue)

    def valueFromUV(self, u: _fnp, v: _fnp, raw=False, rgb_channel=2) -> _fnp:
        """ The texture's value at the given u,v coordinate.

        If raw argument is false, the returned value will be modified according to this texture's value modifier
        function (if any). If raw argument is true, the unmodified value of the texture's source image will be returned.

        :param u: The u coordinate
        :param v: The v coordinate
        :param raw: whether to return the raw value of the source image (otherwise value will be modified by texture's
            value modifier function (if any))
        :param rgb_channel: If the texture is an rgb texture (i.e. has 3 channels), rgb_channel dictates which channel
            the texture value will be taken from. Set rgb_channel to 1, 2 or 3 to use the texture's red, green or blue
            channel respectively. If the texture is not rgb (i.e. has a single channel), the value of rgb_channel is
            ignored.
        :return: the value
        """
        array = self.imageData()
        if array.ndim == 3:
            assert 1 <= rgb_channel <= 3, "rgb_channel must be equal to 1, 2 or 3"
            array = array[:, :, rgb_channel - 1]
        rows, columns = array.shape[:2]
        retShape = None
        if type(u) is np.ndarray:
            retShape = u.shape
        elif type(v) is np.ndarray:
            retShape = v.shape
        if retShape is not None:
            # u and or v are numpy arrays, so multiple texture values are being queried
            ret = np.full(retShape, np.nan, dtype=float)
            noVal = np.isnan(u) + np.isnan(v)
            ret[~noVal] = array[(rows * (v[~noVal] % 1)).astype(int), (columns * (u[~noVal] % 1)).astype(int)]
            if raw:
                return ret
            return self._modifier(ret)
        else:
            # neither u nor v is a numpy array, so only a single texture value is being queried
            if raw:
                return array[int(rows * v), int(columns * u)]
            return self._modifier(array[int(rows * v), int(columns * u)])


class TexturePCS(Texture):
    """Base class for textures which represent a surface using an angles-based (i.e. longitude and latitude) planetary
    coordinate system (PCS).

    Do not use instances of this class, instead use its subclasses (e.g. Planetocentric).
    """
    def __init__(self, image: Union[str, np.ndarray], extent_lims: _lims, frame=Frame.world()):
        """ Initialises a new planetary coordinate system (PCS) texture.

        :param image: the image (either a 2D numpy array, or a path to the image file)
        :param extent_lims: (longwest, longeast, minlat, maxlat) tuple of texture's extent in PCS [degrees].
        :param frame: the PCS's coordinate frame
        """
        super().__init__(image)
        self.longwest = extent_lims[0]  # degrees
        self.longeast = extent_lims[1]  # degrees
        self.latmin = extent_lims[2]  # degrees
        self.latmax = extent_lims[3]  # degrees
        self.metadataPath: Optional[str] = None
        self._frame = frame

    @classmethod
    def fromMetadataTextFile(cls, path: str):
        """Returns a PCS texture initialised from a metadata text file at the given path"""
        metadata = planData.PlanetMetadata(path)
        longwest = metadata.westlong
        longeast = metadata.eastlong
        latmin = metadata.minlat
        latmax = metadata.maxlat
        extentLims = (longwest, longeast, latmin, latmax)
        imPath = metadata.path
        texture = cls(imPath, extentLims)
        texture.metadataPath = path
        return texture

    def latLongCoordFromXYZ(self, point: Vec3):
        """ Converts the given xyz coordinate (in the world frame) to a longitude latitude coordinate in this
        PCS texture's frame.

        :param point: the 3d point (in world coordinates)
        :return: longitude latitude coordinate tuple
        """
        raise NotImplemented

    def uvCoordFromLatLong(self, long: _fnp, lat: _fnp, degrees=True, clip=True) -> Tuple[_fnp, _fnp]:
        """ For a given long-lat coord, this function returns the corresponding u-v image
        coordinate on this PCS texture's image.

        :param long: longitude coordinate.
        :param lat: latitude coordinate.
        :param degrees: whether the given coordinates are in degrees (if not, they are treated as radians).
        :param clip: whether to clip the returned uv coordinate to the range 0-1.
        :return: The image u-v coordinate of the given long-lat coordinate.
        """
        raise NotImplemented

    def uvCoordFromXYZ(self, point: Vec3, clip=True):
        """ For a given xyz coord (in the world frame) this function returns the corresponding u-v image
        coordinate on this PCS texture's image.

        :param point: the 3d point (in the world frame)
        :param clip: whether to clip the resulting u and v values to the 0-1 range
        :return: uv coordinate tuple
        """
        long, lat = self.latLongCoordFromXYZ(point)
        return self.uvCoordFromLatLong(long, lat, degrees=False, clip=clip)

    def valueFromLatLong(self, long: _fnp, lat: _fnp, degrees=True, raw=False, rgb_channel=2) -> Optional[_fnp]:
        """ Return's the texture's image's value at the given longitude and latitude coord.

        If raw argument is false, the returned value will be modified according to this texture's value modifier
        function (if any). If raw argument is true, the unmodified value of the texture's source image will be returned.

        :param long: Longitude coord
        :param lat: Latitude coord
        :param degrees: whether the long and lat coords are in degrees (otherwise treated as radians)
        :param raw: whether to return the raw value of the source image (otherwise value will be modified by texture's
            value modifier function (if any))
        :param rgb_channel: If the texture is an rgb texture (i.e. has 3 channels), rgb_channel dictates which channel
            the texture value will be taken from. Set rgb_channel to 1, 2 or 3 to use the texture's red, green or blue
            channel respectively. If the texture is not rgb (i.e. has a single channel), the value of rgb_channel is
            ignored.
        :return: the texture's value
        """
        u, v = self.uvCoordFromLatLong(long, lat, degrees=degrees)
        if u is None or v is None:
            return None
        return self.valueFromUV(u, v, raw=raw, rgb_channel=rgb_channel)

    def valueFromXYZ(self, point: Vec3, raw=False, rgb_channel=2):
        """ Returns the value of this texture at the given point (in world frame).

        If raw argument is false, the returned value will be modified according to this texture's value modifier
        function (if any). If raw argument is true, the unmodified value of the texture's source image will be returned.

        :param point: the point
        :param raw: whether to return the raw value of the source image (otherwise value will be modified by texture's
            value modifier function (if any))
        :param rgb_channel: If the texture is an rgb texture (i.e. has 3 channels), rgb_channel dictates which channel
            the texture value will be taken from. Set rgb_channel to 1, 2 or 3 to use the texture's red, green or blue
            channel respectively. If the texture is not rgb (i.e. has a single channel), the value of rgb_channel is
            ignored.
        :return: the texture's value
        """
        u, v = self.uvCoordFromXYZ(point)
        return self.valueFromUV(u, v, raw=raw, rgb_channel=rgb_channel)


class TexturePlanetocentric(TexturePCS):
    """ Class for a texture represented in planetocentric coordinates.
    A planetocentric texture maps longitude to texture u coordinate and latitude to texture v
    coordinate, where longitude=0 map to u=0, longitude 360 maps to u=1, latitude=90 maps to v=0, latitude=-90 maps to
    v=1.
    """

    @classmethod
    def fromMetadataTextFile(cls, path: str):
        """Returns a Planetocentric texture initialised from a metadata text file at the given path"""
        metadata = planData.PlanetMetadata(path)
        longwest = metadata.westlong
        longeast = metadata.eastlong
        latmin = metadata.minlat
        latmax = metadata.maxlat
        extentLims = (longwest, longeast, latmin, latmax)
        imPath = metadata.path
        texture = cls(imPath, extentLims)
        texture.metadataPath = path
        return texture

    def latLongCoordFromXYZ(self, point: Vec3):
        """ Converts the given xyz coordinate (in the world frame) to a longitude latitude coordinate in this
        planetocentric texture's frame.

        :param point: the 3d point (in world coordinates)
        :return: longitude latitude coordinate tuple
        """
        _, az, pol = self._frame.fromWorldToSpherical(point)
        lat = 0.5 * np.pi - pol
        return az, lat

    def uvCoordFromLatLong(self, long: _fnp, lat: _fnp, degrees=True, clip=True) -> Tuple[_fnp, _fnp]:
        """ For a given long-lat coord, this function returns the corresponding u-v image
        coordinate on this planetocentric texture's image.

        :param long: longitude coordinate.
        :param lat: latitude coordinate.
        :param degrees: whether the given coordinates are in degrees (if not, they are treated as radians).
        :param clip: whether to clip the returned uv coordinate to the range 0-1.
        :return: The image u-v coordinate of the given long-lat coordinate.
        """
        if not degrees:
            long = np.degrees(long)
            lat = np.degrees(lat)
        long = long % 360
        westLong = self.longwest
        eastLong = self.longeast
        minLat = self.latmin
        maxLat = self.latmax
        if westLong > eastLong:
            #  ori straddles line of zero longitude
            westLong = westLong - 360
            long = long - 360
        u = (long - westLong) / (eastLong - westLong)
        v = (maxLat - lat) / (maxLat - minLat)
        if clip:
            out = (u < 0) + (u >= 1) + (v < 0) + (v >= 1)
            u[out] = np.nan
            v[out] = np.nan
        return u, v


textureType = Union[Texture, TexturePlanetocentric]
