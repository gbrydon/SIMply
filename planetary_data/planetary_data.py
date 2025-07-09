# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
import gc
import numpy as np
import coremaths.geometry
from coremaths.frame import Frame
from coremaths.geometry import Spheroid
import rendering.meshes as meshes
import cv2
from typing import List, Optional, Tuple, Union

_fnp = Union[float, np.ndarray]
_ints = List[int]
_floats = List[float]
_radii = Union[float, Tuple[float, float, float]]


class PlanetMetadata:
    """Class with basic functionality for storing/reading/writing metadata describing a planetary data product
    (e.g. DTM/ORI)"""

    # metadata keys:
    _westlongKey = "longwest"
    _eastlongKey = "longeast"
    _minlatKey = "latmin"
    _maxlatKey = "latmax"
    _vminKey = "vmin"
    _vmaxKey = "vmax"
    _columnsKey = "columns"
    _rowsKey = "rows"
    _sphRadXKey = "sphRadX"
    _sphRadYKey = "sphRadY"
    _sphRadZKey = "sphRadZ"
    _xresKey = "xres"
    _yresKey = "yres"
    _pathKey = "path"
    _sourceKey = "source"
    _dtypeStringKey = "dtype"

    def __init__(self, path: Optional[str] = None):
        """ Initialises a new planetary data metadata object.
        If a path to saved metadata is provided, this metadata object will be initialised with those metadata, otherwise
        it will be empty.

        :param path: optional path to saved metadata (see class's loadFromFile function for info on file format)
        """
        self._metadataDict = dict()
        if path is not None:
            self.loadFromFile(path)

    @classmethod
    def fromIMG(cls, path: str):
        """ Returns a PlanetMetadata object whose metadata has been loaded from the given PDS-format .IMG planetary data
         product.

        :param path: path to the .IMG file
        :return: metadata object containing the file's metadata
        """
        ret = cls()
        ret.readFrom(path, ".IMG")
        return ret

    @classmethod
    def metadataType(cls, key: str) -> str:
        """ For a given key this function returns a string indicating the type of metadata associated with that
        key

        :param key: the key
        :return: the data type associated with the key (e.g. int/float/str)
        """
        intKeys = {cls._rowsKey, cls._columnsKey}
        floatKeys = {cls._westlongKey, cls._eastlongKey, cls._minlatKey, cls._maxlatKey, cls._vminKey, cls._vmaxKey,
                     cls._sphRadXKey, cls._sphRadYKey, cls._sphRadZKey, cls._xresKey, cls._yresKey}
        strKeys = {cls._pathKey, cls._sourceKey, cls._dtypeStringKey}
        if key in intKeys:
            return 'int'
        elif key in floatKeys:
            return 'float'
        elif key in strKeys:
            return 'str'
        else:
            raise KeyError('metadata file contained an unexpected (key,value) pair with key "{}"'.format(key))

    @property
    def westlong(self):
        """Westernmost longitude of planetary data product [degrees]"""
        return self._metadataDict[self._westlongKey]

    @property
    def eastlong(self):
        """Easternmost longitude of planetary data product [degrees]"""
        return self._metadataDict[self._eastlongKey]

    @property
    def minlat(self):
        """Minimum latitude of planetary data product [degrees]"""
        return self._metadataDict[self._minlatKey]

    @property
    def maxlat(self):
        """Maximum latitude of planetary data product [degrees]"""
        return self._metadataDict[self._maxlatKey]

    @property
    def vmin(self) -> Optional[float]:
        """Minimum valid value of data"""
        try:
            return self._metadataDict[self._vminKey]
        except KeyError:
            return None

    @property
    def vmax(self) -> Optional[float]:
        """Maximum valid value of data"""
        try:
            return self._metadataDict[self._vmaxKey]
        except KeyError:
            return None

    @property
    def columns(self) -> int:
        """Number of columns in data product"""
        return self._metadataDict[self._columnsKey]

    @property
    def rows(self) -> int:
        """Number of rows in data product"""
        return self._metadataDict[self._rowsKey]

    @property
    def sphRadX(self):
        """Data product's datum spheroid's radius in x direction [m]."""
        return self._metadataDict[self._sphRadXKey]

    @property
    def sphRadY(self):
        """Data product's datum spheroid's radius in y direction [m]."""
        return self._metadataDict[self._sphRadYKey]

    @property
    def sphRadZ(self):
        """Data product's datum spheroid's radius in z direction [m]."""
        return self._metadataDict[self._sphRadZKey]

    @property
    def xres(self):
        return self._metadataDict[self._xresKey]

    @property
    def yres(self):
        return self._metadataDict[self._yresKey]

    @property
    def path(self):
        """Path to the planetary data product"""
        return self._metadataDict[self._pathKey]

    @property
    def source(self):
        """Source of the planetary data"""
        return self._metadataDict[self._sourceKey]

    @property
    def dtype(self):
        """Data type of data in the planetary data product"""
        return self._metadataDict[self._dtypeStringKey]

    @dtype.setter
    def dtype(self, value: str):
        self._metadataDict[self._dtypeStringKey] = value

    def loadFromFile(self, path: str):
        """ Loads a metadata object from file. The given path should point to a text file containing labelled metadata
        as key:value pairs.
        There should be one key:value pair per line. Key and value should be separated by a colon (":").

        See PlanetMetadata class for supported metadata values and their keys.

        :param path: path to the metadata text file.
        """
        t = open(path, 'rt')
        for line in t:
            key, value = line.strip().split(':')
            valueType = self.metadataType(key)
            if valueType == 'int':
                self._metadataDict[key] = int(value.strip())
            elif valueType == 'float':
                self._metadataDict[key] = float(value.strip())
            elif valueType == 'str':
                self._metadataDict[key] = str(value.strip())

    def saveToFile(self, path: str):
        """ Writes this metadata to a metadata text file at the specified path.

        The metadata are written into a .txt file where each line contains a single key: value pair, in which the key
        and value are separated by a colon (":").
        """
        with open(path, 'w') as f:
            for key, value in self._metadataDict.items():
                f.write('{0}: {1}\n'.format(key, value))
        f.close()

    def readFrom(self, path: str, source: str):
        """ Reads metadata from the file at the given path with the given source type (either '.IMG', '.LBL' or
        'geotfif') into this metadata object.

        :param path: path to the file containing the metatdata
        :param source: source string identifying the type of file the metadata is being read from
        """
        if source == '.IMG' or source == '.LBL':
            self._readTextMetadataPDS(path)
        elif source[:6] == 'geotif':
            self._readFromWGS84Geotiff(path)
        else:
            raise TypeError('Invalid/unsupported source ({}) of planetary data.'.format(source))

    def _readTextMetadataPDS(self, path: str):
        """ Reads the text-based PDS-format metadata for planetary surface image data (ORI or DTM) from the file at the
        given path (this could be an .IMG data product or a .LBL file) into this metadata object.

        :param path: path to the file containing the text-based metadata information in PDS format
        """
        t = open(path, 'rt')
        metadata = dict()
        metadata[self._sourceKey] = '.IMG'
        metadata[self._pathKey] = path
        dataTypeStr = None
        dataBits = None
        while True:
            line = t.readline()
            words = line.split()
            if len(words) == 0:
                continue
            if words[0] == "LINES":
                metadata[self._rowsKey] = int(words[-1])
                continue
            if words[0] == "LINE_SAMPLES":
                metadata[self._columnsKey] = int(words[-1])
                continue
            if words[0] == "VALID_MINIMUM":
                metadata[self._vminKey] = float(words[-1])
                continue
            if words[0] == "VALID_MAXIMUM":
                metadata[self._vmaxKey] = float(words[-1])
                continue
            if words[0] == "MINIMUM_LATITUDE":
                metadata[self._minlatKey] = float(words[-2])
                continue
            if words[0] == "MAXIMUM_LATITUDE":
                metadata[self._maxlatKey] = float(words[-2])
                continue
            if words[0] == "WESTERNMOST_LONGITUDE":
                metadata[self._westlongKey] = float(words[-2])
                continue
            if words[0] == "EASTERNMOST_LONGITUDE":
                metadata[self._eastlongKey] = float(words[-2])
                continue
            if words[0] == "A_AXIS_RADIUS":
                metadata[self._sphRadXKey] = 1000 * float(words[-2])
                continue
            if words[0] == "B_AXIS_RADIUS":
                metadata[self._sphRadYKey] = 1000 * float(words[-2])
                continue
            if words[0] == "C_AXIS_RADIUS":
                metadata[self._sphRadZKey] = 1000 * float(words[-2])
                continue
            if words[0] == "SAMPLE_TYPE":
                dataTypeStr = str(words[-1])
                continue
            if words[0] == "SAMPLE_BITS":
                dataBits = int(words[-1])
                continue
            if line == "END\n":
                break
        t.close()

        if (metadata[self._eastlongKey] % 360) < (metadata[self._westlongKey] % 360):
            xres = (360 + metadata[self._eastlongKey] - metadata[self._westlongKey]) / metadata[self._columnsKey]
            metadata[self._xresKey] = xres
        else:
            xres = (metadata[self._eastlongKey] - metadata[self._westlongKey]) / metadata[self._columnsKey]
            metadata[self._xresKey] = xres
        metadata[self._yresKey] = (metadata[self._maxlatKey] - metadata[self._minlatKey]) / metadata[self._rowsKey]

        if dataTypeStr is not None and dataBits is not None:
            if "int" in dataTypeStr.lower():
                metadata[self._dtypeStringKey] = "<i{}".format(int(dataBits / 8))
            elif "ieee_real" in dataTypeStr.lower():
                metadata[self._dtypeStringKey] = ">f{}".format(int(dataBits / 8))
            else:
                metadata[self._dtypeStringKey] = "<f{}".format(int(dataBits / 8))

        self._metadataDict = metadata

    def _readFromWGS84Geotiff(self, path: str):
        """Reads the metadata from the WGS84 geotif at the given path into this metadata object"""
        try:
            from osgeo import gdal
        except ImportError:
            gdal = None
        if gdal is None:
            print("install gdal (https://pypi.org/project/GDAL/) to work with geotif files")
            return

        product = gdal.Open(path)
        metadata = dict()
        metadata['source'] = 'geotiff wgs84'
        metadata['path'] = path
        metadata['columns'] = product.RasterXSize
        metadata['rows'] = product.RasterYSize
        metadata['vmin'], metadata['vmax'], _, _ = product.GetRasterBand(1).GetStatistics(True, True)
        metadata['longwest'], xres, _, metadata['latmax'], _, yres = product.GetGeoTransform()
        metadata['latmin'] = metadata['latmax'] + (metadata['rows'] * yres)
        metadata['longeast'] = metadata['longwest'] + (metadata['columns'] * xres)
        metadata['sphRadX'] = 6378137  # WGS84 datum
        metadata['sphRadY'] = 6378137  # WGS84 datum
        metadata['sphRadZ'] = 6356752  # WGS84 datum
        self._metadataDict = metadata


class PlanetDTM:
    """Class for working with planetary DTMs/DEMs."""
    def __init__(self, metadata_path: Optional[str] = None, dtm_path: Optional[str] = None,
                 dtm_source: Optional[str] = None):
        """ Initialises a new PlanetDTM, either from a metadata file at the given path, or from a DTM at the given path
        and with the given source

        :param metadata_path: path to text metadata file detailing DTM (needed if no dtm_path & dtm_source are provided;
            see PlanetMetadata.loadFromFile() for details of file's required format)
        :param dtm_path: path to DTM file (needed if no metadata_path is provided)
        :param dtm_source: string id of DTM source (".IMG"/".LBL"/"geotif") (needed if no metadata_path is provided)
        """
        self._metadata = PlanetMetadata(metadata_path)
        if metadata_path is None:
            if dtm_path is not None and dtm_source is not None:
                self._metadata.readFrom(dtm_path, dtm_source)
        self._elevationMultiplier = 1

    @property
    def metadata(self) -> PlanetMetadata:
        """This DTM's metadata as a PlanetMetadata object."""
        return self._metadata

    @property
    def elevationData(self) -> Optional[np.ndarray]:
        """ Reads and returns the elevation values stored in the DTM.

        If the DTM is an .IMG file it is important to ensure the dtype string in the metadata is correct to ensure the
        elevation data are read correctly (the DTM's data type should be stated in the .IMG file's header, and can be
        checked by opening it with a text editor).

        :return: the DTM's elevation data as a 2D numpy array.
        """
        if self._metadata.source == '.IMG':
            return readDataFromIMG(self._metadata.path, self._metadata.dtype) * self._elevationMultiplier
        elif self._metadata.source[:6] == 'geotif':
            try:
                from osgeo import gdal
            except ImportError:
                gdal = None
            if gdal is None:
                print("install gdal (https://pypi.org/project/GDAL/) to work with geotif files")
                return None

            d = gdal.Open(self._metadata.path)
            return d.ReadAsArray().astype('float') * self._elevationMultiplier
        else:
            return cv2.imread(self._metadata.path) * self._elevationMultiplier

    @property
    def datum(self):
        """ The spheroid defining this DTM's datum surface."""
        rx = self.metadata.sphRadX
        ry = self.metadata.sphRadY
        rz = self.metadata.sphRadZ
        return coremaths.geometry.Spheroid(Frame.world(), rx, ry, rz)

    def getPointGrid(self, sub_region=None, dsf=1, save_path: str = None):
        """ Generates and returns a point grid (a 2D array of points in 3D space describing the DTM's pixels' centres)
        from this DTM. The point grid is returned as a Vec3 object with numpy array components.

        If a save path is provided, the point grid is also saved as a numpy array at the given path.

        :param sub_region: optional tuple giving western longitude, eastern longitude, min latitude and max latitude of
            a sub-region of the DTM's coverage, to which the generated point grid will be limited [degrees]. This region
            must sit entirely within the DTM's extent.
            If =None, the point grid is generated from the whole DTM's extent.
        :param dsf: down-sampling factor - integer factor by which the DTM will be downsampled before point grid is
            generated (dsf=1 performs no down-sampling).
        :param save_path: optional path for resulting point grid to be saved as a numpy array.
        :return: the point grid derived from this DTM.
        """
        if dsf < 1:
            raise ValueError("dsf must be an integer greater than or equal to 1, but value of {} was used".format(dsf))

        metadata = self.metadata
        rows = metadata.rows
        columns = metadata.columns
        latmin = metadata.minlat
        latmax = metadata.maxlat
        longwest = metadata.westlong % 360
        longeast = metadata.eastlong % 360
        vmin = metadata.vmin
        vmax = metadata.vmax

        if sub_region is not None:
            sub_region = [sub_region[0] % 360, sub_region[1] % 360, sub_region[2], sub_region[3]]

        if longeast < longwest:
            longeast = longeast + 360
            if sub_region is not None:
                if sub_region[0] < longwest:
                    sub_region[0] = sub_region[0] + 360
                if sub_region[1] < longwest:
                    sub_region[1] = sub_region[1] + 360

        long = np.linspace(longwest, longeast, columns)
        lat = np.linspace(latmax, latmin, rows)

        if sub_region is not None:
            idx_longMin = np.abs(long - sub_region[0]).argmin()
            idx_longMax = np.abs(long - sub_region[1]).argmin()
            idx_latMin = np.abs(lat - sub_region[2]).argmin()
            idx_latMax = np.abs(lat - sub_region[3]).argmin()
            long = long[idx_longMin:idx_longMax + 1]
            lat = lat[idx_latMax:idx_latMin + 1]
            e = self.elevationData[idx_latMax:idx_latMin + 1, idx_longMin:idx_longMax + 1]
        else:
            e = self.elevationData
        if np.issubdtype(e.dtype, np.integer):
            e = e.astype(float)
        e[e < vmin] = np.nan
        e[e > vmax] = np.nan

        if dsf != 1:
            long = long[::dsf]
            lat = lat[::dsf]
            e = e[::dsf, ::dsf]

        long, lat = np.meshgrid(long, lat)

        points = self.convertSurfaceLongLatToXYZ(long, lat, e=e, degrees=True)
        if save_path is not None:
            shape = points.numpyShape + (3,)
            arr = np.empty(shape)
            arr[:, :, 0] = points.x
            arr[:, :, 1] = points.y
            arr[:, :, 2] = points.z
            np.save(save_path, arr)
        return points

    def getTrimesh(self, sub_region=None, dsf=1, save_path: str = None, retain_grid_info=False):
        """ Generates a trimesh from the given DTM and returns it. If a save_path is provided, the trimesh is
            also saved as a pair of numpy arrays (a vertices array and a tris array) at the given path.

        :param sub_region: optional tuple giving western longitude, eastern longitude, min latitude and max latitude of
            a sub-region of the DTM's coverage, to which the generated point grid will be limited. This region must sit
            entirely within the DTM's extent. If =None, the point grid is generated from the whole DTM's extent.
        :param dsf: down-sampling factor - integer factor by which the DTM will be downsampled before trimesh is
                    generated (dsf=1 performs no down-sampling)
        :param save_path: optional path for resulting trimesh to be saved as a numpy arrays (exclude file extension)
        :param retain_grid_info: whether to retain the information describing the connection between the mesh's
            vertices and the grid of points (dtm) from which they're extracted.
        :return: the trimesh
        """
        grid = self.getPointGrid(sub_region=sub_region, dsf=dsf)
        mesh = meshes.Mesh.fromPointGrid(grid.x, grid.y, grid.z)
        if retain_grid_info is False:
            mesh = mesh.meshStrippedOfNans
        if save_path is not None:
            np.save(save_path + '_verts.npy', mesh.vertices)
            np.save(save_path + '_tris.npy', mesh.tris)
        return mesh

    def convertSurfaceLongLatToXYZ(self, long: _fnp, lat: _fnp, e: _fnp = None, degrees=True):
        """ For a planetary surface defined by the given DTM, this function converts a long-lat coord to a xyz coord

        :param long: longitude coord(s).
        :param lat: latitude coord(s).
        :param e: optional elevation value(s) of surface for the given longitude and latitude coords (providing this if
            it has already been calculated will minimise computations performed by this function and improve speed when
            dealing with large DTMs).
        :param degrees: whether long & lat values are in degrees (otherwise treated as radians).
        :return: XYZ coord.
        """
        rows = self.metadata.rows
        columns = self.metadata.columns
        latmin = self.metadata.minlat
        latmax = self.metadata.maxlat
        longwest = self.metadata.westlong
        longeast = self.metadata.eastlong
        imExtent = (longwest, longeast, latmin, latmax)

        vmin = self.metadata.vmin
        vmax = self.metadata.vmax
        ra = self.metadata.sphRadX
        rb = self.metadata.sphRadY
        rc = self.metadata.sphRadZ
        spheroid = Spheroid(Frame.world(), ra, rb, rc)

        if e is None:
            u, v = planetocentricImageUVCoord(long, lat, imExtent, degrees=degrees)
            col = u * columns
            row = v * rows

            if type(long) is np.ndarray or type(lat) is np.ndarray:
                notNan = ~np.isnan(col) * ~np.isnan(row)
                r_index = row[notNan].astype(int)
                c_index = col[notNan].astype(int)
                if type(long) is np.ndarray:
                    shape = long.shape
                else:
                    shape = lat.shape
                e = np.full(shape, np.nan, dtype=float)
                e[notNan] = self.elevationData[r_index, c_index]
                del notNan
                del r_index
                del c_index
                del col
                del row
                gc.collect()
                if vmin is not None:
                    e[e < vmin - 100] = np.nan
                if vmax is not None:
                    e[e > vmax + 100] = np.nan
            else:
                e = self.elevationData[int(row), int(col)]
                if vmin is not None:
                    if e < vmin:
                        e = None
                if vmax is not None:
                    if e > vmax:
                        e = None

        if degrees:
            long = np.radians(long)
            lat = np.radians(lat)
        polar = 0.5 * np.pi - lat
        del lat
        gc.collect()
        radius = spheroid.radius(long, polar)
        points = Frame.world().fromSpherical(radius + e, long, polar)
        del long
        del polar
        del radius
        gc.collect()
        return points

    def coregisterMesh(self, mesh: meshes.Mesh, vert_indices: _ints, longs: _floats, lats: _floats, degrees=True):
        """ Takes a triangular mesh represented in an arbitrary 3D cartesian coordinate frame, the indices of at least 3
        of its vertices, and the longitudes and latitudes of those vertices in this planetary DTM's planetocentric
        coordinate frame, and finds the similarity transform for transforming points from the mesh's coordinate frame
        to this DTM's 3D planetocentric frame.

        The similarity transform is a (R, t, s) tuple (rotation matrix, translation vector, scale factor).
        A point p is converted from this mesh's coordinate frame to the new coordinate frame by pNew = s * R * p + t.

        :param mesh: triangular mesh
        :param vert_indices: list of indices of >=3 of mesh's vertices
        :param longs: list of longitudes of the mesh vertices, in this DTM's planetocentric frame
        :param lats: list of latitudes of the mesh vertices, in this DTM's planetocentric frame
        :param degrees: whether long & lat values are in degrees (otherwise treated as radians)
        :return: tuple containing te similarity transform used to perform the conversion
        """
        planetocentricCoords = []
        for long, lat, vertIndex in zip(longs, lats, vert_indices):
            planetocentricCoords += [self.convertSurfaceLongLatToXYZ(long, lat, degrees=degrees)]
        return mesh.similarityTransformation(vert_indices, planetocentricCoords)


def readDataFromIMG(path: str, dtype: str) -> np.ndarray:
    """ Given the path to a 2D planetary data image product in .IMG file format, this function reads and returns the
    image data.

    :param path: path to the .IMG file
    :param dtype: string indicating the datatype in the file (see https://numpy.org/devdocs/user/basics.types.html and
        https://www.w3schools.com/python/numpy/numpy_data_types.asp)
    :return: the image data
    """
    metadata = PlanetMetadata.fromIMG(path)
    cols = metadata.columns
    rows = metadata.rows
    fid = open(path, 'rb')
    data = np.fromfile(fid, dtype=dtype)
    data = data[-cols * rows:]
    data = data.reshape((rows, cols))
    return data


def planetocentricImageUVCoord(long: _fnp, lat: _fnp, im_extent: Tuple[float, float, float, float], degrees=True):
    """ Converts long-lat coord to image UV coord for a planetocentric data product with given longitude and latitude
    extents.

    :param long: longitude coord
    :param lat: latitude coord
    :param im_extent: data product extent as (longwest, longeast, minlat, maxlat) tuple
    :param degrees: whether the given longitudes and latitudes are in degrees (otherwise treated as radians)
    :return: tuple containing u,v coord
    """
    if not degrees:
        long = np.degrees(long)
        lat = np.degrees(lat)
    long = long % 360

    westLong = im_extent[0]
    eastLong = im_extent[1]
    minLat = im_extent[2]
    maxLat = im_extent[3]
    if not degrees:
        westLong = np.degrees(westLong)
        eastLong = np.degrees(eastLong)
        minLat = np.degrees(minLat)
        maxLat = np.degrees(maxLat)
    westLong = westLong % 360
    eastLong = eastLong % 360
    if westLong > eastLong:
        #  image straddles meridian of zero longitude
        westLong = westLong - 360
        long = long - 360

    u = (long - westLong) / (eastLong - westLong)
    v = (maxLat - lat) / (maxLat - minLat)
    out = (u < 0) + (u >= 1) + (v < 0) + (v >= 1)
    if type(u) is np.ndarray:
        u[out] = np.nan
    if type(v) is np.ndarray:
        v[out] = np.nan
    return u, v


def convertSurfLongLatElToXYZ(long: _fnp, lat: _fnp, el: _fnp, surf_rad: _radii, degrees=True):
    if degrees:
        long = np.radians(long)
        lat = np.radians(lat)

    if type(surf_rad) is tuple or type(surf_rad) is list:
        a, b, c = surf_rad
    else:
        a, b, c = surf_rad, surf_rad, surf_rad
    spheroid = Spheroid(Frame.world(), a, b, c)

    localRadius = spheroid.radius(long, 0.5 * np.pi - lat)

    return Frame.fromSpherical(localRadius + el, long, 0.5 * np.pi - lat)
