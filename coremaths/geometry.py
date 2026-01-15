# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""
Module containing classes describing various geometric shapes.
"""

import numpy as np
from coremaths import math2
from coremaths.vector import Vec2, Vec3
from coremaths.frame import Frame
from coremaths.ray import Ray
import warnings
from typing import Dict, List, Optional, Tuple, Union

_fnp = Union[float, np.ndarray]
_inp = Union[int, np.ndarray]


class Geometry:
    """Base class for a geometry object. Do not use this base class directly, instead use subclasses."""
    def __init__(self, frame: 'Frame' = Frame.world()):
        """Initialises an instance of the geometry base class.
        This base class should not be used, instead use subclasses.

        :param frame: the frame describing the position and orientation of the geometry in 3d space
        """
        self._frame: 'Frame' = frame

    @staticmethod
    def rectangle(frame: 'Frame', lx: _fnp, ly: _fnp):
        """Returns a 2D rectangle geometry in 3D space.

        :param frame: The rectangle's frame. The origin of the frame is the centre of the rectangle. The w axis of the
         frame defines the normal of the rectangle's surface. The u and v axes define the x and y axes of the rectangle
         respectively.
        :param lx: The rectangle's length along its x-axis.
        :param ly: The rectangle's length along its y-axis.
        :return: The rectangle geometry.
        """
        return Rectangle(frame, lx, ly)

    @staticmethod
    def cuboid(frame: 'Frame', lx: _fnp, ly: _fnp, lz: _fnp):
        """Returns a cuboid geometry of the given size.

        :param frame: The cuboid's frame. This frame's origin defines the centre of the cuboid. The x, y and z faces of
        the cube are each centred and perpendicular to the x, y and z axes of the frame respectively.
        :param lx: Cuboid length along its x-axis
        :param ly: Cuboid length along its y-axis
        :param lz: Cuboid length along its z axis
        :return: The cuboid.
        """
        return Cuboid(frame, lx, ly, lz)

    @staticmethod
    def spheroid(frame: 'Frame', rx: _fnp, ry: _fnp, rz: _fnp):
        """Returns a spheroid geometry

        :param frame: The spheroid's frame. This frame's origin defines the centre of the spheroid.
        :param rx: Spheroid radius [m] measured along its x-axis
        :param ry: Spheroid radius [m] measured along its y-axis
        :param rz: Spheroid radius [m] measured along its z-axis
        :return: The spheroid.
        """
        return Spheroid(frame, rx, ry, rz)

    @property
    def frame(self) -> 'Frame':
        """The geometry's frame, describing its position and orientation in 3d space"""
        return self._frame

    @frame.setter
    def frame(self, new: 'Frame'):
        """Updates the geometry's frame, which describes its position and orientation in 3D space."""
        self._frame = new

    @property
    def isNumpyType(self) -> bool:
        """Whether any of this geometry's properties are represented by numpy arrays."""
        return self._frame.isNumpyType

    def normal(self, point: 'Vec3') -> 'Vec3':
        """Returns the outward-pointing normal of this geometry's surface at the given point.

        :param point: The point (in world frame) on the surface where the normal is required. This point should
                    be on or very close to the surface. Providing a point that is far from the surface may
                    give meaningless results.
        :return: The surface normal (in the world frame) at the given point
        """
        message = "function not implemented - use a subclass of Geometry with an implemented normal(...) function"
        warnings.warn(message)
        return Vec3.zero()

    def uv(self, point: Vec3) -> Optional[Tuple[_fnp, _fnp]]:
        """ Returns the uv coord for the given point on this geometry's surface.

        :param point: The point (in world frame) on the surface. This point should be on or very close to the surface,
                    otherwise the returned uv coordinate may be meaningless.
        :return: u,v coordinate tuple.
        """
        message = "function not implemented - use a subclass of Geometry with an implemented uv() function"
        warnings.warn(message)
        return None

    def pointFromUV(self, u: _fnp, v: _fnp) -> Optional[Vec3]:
        """ Returns the 3d coordinate (in the world frame) of the point on this geometry's surface with the given
        uv texture coordinate.

        :param u: u texture coordinate
        :param v: v texture coordinate
        :return: 3D coordinate of point on surface (in world frame)
        """
        message = "function not implemented - use a subclass of Geometry with an implemented pointFromUV() function"
        warnings.warn(message)
        return None

    def intersect(self, ray: 'Ray', epsilon=1e-7, max_dist: float = 1e15, shift: Vec3 = None) -> Dict[str, np.ndarray]:
        """ Calculates and returns the depth and position of the first positive intersection of the given ray with this
        geometry, if any.

        The returned result is a dictionary containing the hit depths ("t_hit"), hit surface normals
        ("primitive_normals") and hit geometry uv coordinates ("primitive_uvs").

        :param ray: The ray to test for intersection with the geometry (must be in world coordinate frame)
        :param epsilon: Values below this epsilon are considered equal to 0 when checking for e.g. parallel geometries
        :param max_dist:  Maximum distance of intersection. Any intersections beyond this distance are discarded.
        :param shift: Vector by which the geometry is shifted before performing intersection test (geometry is shifted
            back after intersection test). Can be used to shift the geometry closer to the origin (if the input rays
            have been shifted by the same vector) to improve ray tracing accuracy by reducing floating point errors.
        :return: Dictionary of the ray intersection result
        """
        #  This function should be overridden by all subclasses of Geometry.
        message = "function not implemented - use a subclass of Geometry with an implemented intersect(...) function"
        warnings.warn(message)
        return {}


class Plane(Geometry):
    """A 2D infinite plane in 3D space."""
    def __init__(self, frame: 'Frame'):
        """ Initialises a new 2D plane of infinite extent in 3D space.
        The plane's orientation in space is dictated by the given frame.

        :param frame: A 3D frame whose w axis is the plane's normal, and whose origin lies on the plane
        """
        super().__init__(frame)

    @classmethod
    def parallelTo(cls, v1: Vec3, v2: Vec3, origin: Vec3):
        """ Returns a plane which is parallel to both v1 and v2, with the given origin point lying on the plane.

        The normal direction of the returned plane is given by the cross product of v1 with v2.

        :param v1: vector parallel to the plane.
        :param v2: another vector parallel to the plane.
        :param origin: origin of the plane's coordinate frame. This point lies on the returned plane.
        :return: the plane.
        """
        w = v1.cross(v2).norm
        frame = Frame.withW(w, origin=origin)
        return cls(frame)

    @property
    def n(self) -> 'Vec3':
        """The plane's normal"""
        return self._frame.w

    def normal(self, point: 'Vec3') -> 'Vec3':
        return self.n

    def perpDistTo(self, point: Vec3) -> _fnp:
        """ Returns the shortest (i.e. perpendicular) distance from the given point to this plane.

        :param point: the point.
        :return: the point's perpendicular distance from the plane.
        """
        return (point - self.frame.origin).dot(self.n)

    def uv(self, point: Vec3) -> Optional[Tuple[_fnp, _fnp]]:
        warnings.warn("uv texture coordinate for an infinite plane does not exist.")
        return None

    def pointFromUV(self, u: _fnp, v: _fnp) -> Optional[Vec3]:
        warnings.warn("uv texture coordinate for an infinite plane does not exist.")
        return None

    def intersect(self, ray: 'Ray', epsilon=1e-7, max_dist: float = 1e15, shift: Vec3 = None) -> Dict[str, np.ndarray]:
        if shift is not None:
            self.frame = self.frame.translated(shift)
        ret = {}
        denominator = self.n.dot(ray.d)
        if self.isNumpyType or ray.isNumpyType:
            depth = np.full_like(denominator, np.nan, dtype=float)
            c1 = np.abs(denominator) > epsilon
            t = (self._frame.origin.npMask(c1) - ray.origin.npMask(c1)).dot(self.n.npMask(c1)) / denominator[c1]
            c2 = t >= 0
            c1[c1] = c1[c1] * c2
            depth[c1] = t[c2]
            inRange = depth <= max_dist
            depth[~inRange] = np.nan
            ret['t_hit'] = depth
            n = self.n * ray.d.norm.length  # done for if only one of ray and normal is numpy type to make both numpy
            n.x[~c1 + ~inRange] = np.nan
            n.y[~c1 + ~inRange] = np.nan
            n.z[~c1 + ~inRange] = np.nan
            ret['primitive_normals'] = n.asNumpyArray
        else:
            if abs(denominator) > epsilon:
                depth = (self._frame.origin - ray.origin).dot(self.n) / denominator
                if depth > max_dist or depth < 0:
                    return {}
                ret['t_hit'] = np.array(depth)
                ret['primitive_normals'] = np.array((self.n.x, self.n.y, self.n.z))
            ret = {}
        if shift is not None:
            self.frame = self.frame.translated(-shift)
        return ret


class Closed2D(Plane):
    """Base class for a closed 2D shape in 3D space.

    Do not use this class directly, instead use subclasses of it.
    """
    @property
    def centre(self) -> Vec3:
        """The geometric centre of the shape."""
        return self.frame.origin

    @property
    def area(self) -> '_fnp':
        """The one-sided surface area of the shape."""
        message = "area property not implemented - use a subclass of Closed2D with an implemented area property."
        warnings.warn(message)
        return 0

    def uv(self, point: Vec3) -> Optional[Tuple[_fnp, _fnp]]:
        message = "function not implemented - use a subclass of Closed2D with an implemented uv(...) function"
        warnings.warn(message)
        return None

    def pointFromUV(self, u: _fnp, v: _fnp) -> Optional[Vec3]:
        message = "function not implemented - use a subclass of Closed2D with implemented pointFromUV(...) function"
        warnings.warn(message)
        return None


class Polygon(Closed2D):
    """Base class for a closed polygon (a 2D shape with a closed boundary comprising a number of connected straight
    lines) in 3D space.

    Do not use this class directly, instead use subclasses of it."""
    @property
    def vertices(self) -> List['Vec3']:
        """An ordered list of all the polygon's vertices' coordinates."""
        message = "vertices property not implemented - use a subclass of Polygon with implemented vertices property"
        warnings.warn(message)
        return []


class Rectangle(Polygon):
    """A 2D rectangle in 3D space."""
    def __init__(self, frame: 'Frame', lx: _fnp, ly: _fnp):
        """ Initialises a 2D rectangle geometry in 3D space.

        :param frame: The rectangle's frame. The origin of the frame is the centre of the rectangle. The w axis of the
                    frame defines the normal of the rectangle's surface. The u and v axes define the x and y axes of the
                    rectangle respectively.
        :param lx: The rectangle's length along its x-axis.
        :param ly: The rectangle's length along its y-axis.
        """
        self.lx = lx
        self.ly = ly
        super().__init__(frame)

    @property
    def area(self) -> '_fnp':
        """The one-sided surface area of the rectangle."""
        return self.lx * self.ly

    @property
    def vertices(self) -> List[Vec3]:
        c1 = self.frame.toWorld(0.5 * Vec3((self.lx, self.ly, 0)))
        c2 = self.frame.toWorld(0.5 * Vec3((-self.lx, self.ly, 0)))
        c3 = self.frame.toWorld(0.5 * Vec3((-self.lx, -self.ly, 0)))
        c4 = self.frame.toWorld(0.5 * Vec3((self.lx, -self.ly, 0)))
        return [c1, c2, c3, c4]

    @property
    def isNumpyType(self) -> bool:
        return type(self.lx) is np.ndarray or type(self.ly) is np.ndarray or super().isNumpyType

    def uv(self, point: Vec3) -> Optional[Tuple[_fnp, _fnp]]:
        plocal = self._frame.fromWorld(point)
        u = plocal.x / self.lx + 0.5
        v = plocal.y / self.ly + 0.5
        return u, v

    def pointFromUV(self, u: _fnp, v: _fnp) -> Optional[Vec3]:
        x = (u - 0.5) * self.lx
        y = (v - 0.5) * self.ly
        return self.frame.toWorld(Vec3((x, y, 0)))

    def intersect(self, ray: 'Ray', epsilon=1e-7, max_dist: float = 1e15, shift: Vec3 = None) -> Dict[str, np.ndarray]:
        ret = super().intersect(ray, epsilon=epsilon, shift=shift)
        if shift is not None:
            self.frame = self.frame.translated(shift)
        if ray.isNumpyType or self.isNumpyType:
            d = ret['t_hit']
            p = ray.point(d)
            plocal = self._frame.fromWorld(p)
            c1 = abs(plocal.x) <= 0.5 * self.lx
            c2 = abs(plocal.y) <= 0.5 * self.ly
            d[~(c1 * c2)] = np.nan
            ret['t_hit'] = d
            n = ret['primitive_normals']
            n[~(c1 * c2)] = np.nan
            ret['primitive_normals'] = n
            u = plocal.x / self.lx + 0.5
            v = plocal.y / self.ly + 0.5
            u[~(c1 * c2)] = np.nan
            v[~(c1 * c2)] = np.nan
            uv = Vec2((u, v))
            ret['primitive_uvs'] = uv.asNumpyArray
        else:
            d = ret['t_hit']
            p = ray.point(d)
            plocal = self._frame.fromWorld(p)
            c1 = abs(plocal.x) <= 0.5 * self.lx
            c2 = abs(plocal.y) <= 0.5 * self.ly
            if c1 and c2:
                u = plocal.x / self.lx + 0.5
                v = plocal.y / self.ly + 0.5
                ret['primitive_uvs'] = np.array((u, v))
            else:
                ret = {}

        if shift is not None:
            self.frame = self.frame.translated(-shift)

        return ret


class Polyhedron(Geometry):
    """Base class for a polyhedron (a 3D solid made up of polygons joined at their edges).

    Do not use this class directly, instead use subclasses of it.
    """
    @property
    def faces(self) -> List[Polygon]:
        """An ordered list of all the polyhedron's faces."""
        message = "faces property not implemented - use a subclass of Polyhedron with implemented faces property"
        warnings.warn(message)
        return []

    def faceFromSurfacePoint(self, point: Vec3) -> _inp:
        """ Returns the index of the face of this polyhedron that is closest to the given point (in world coords)

        :param point: the point, in world coordinates (this should be on or very close to a face, otherwise the result
                    will be inaccurate)
        :return: the index of the closest face
        """
        message = "function not implemented - use a subclass of Polyhedron with implemented faceFromSurfacePoint(...)"
        warnings.warn(message)
        return 0

    def faceFromUV(self, u: _fnp, v: _fnp) -> Optional[_inp]:
        """ For the given uv texture coordinate for this polyhedron, this function returns the corresponding face of the
         polyhedron as an index

        :param u: u texture coordinate
        :param v: v texture coordinate
        :return: face index
        """
        message = "function not implemented - use a subclass of Polyhedron with implemented faceFromUV(...) function"
        warnings.warn(message)
        return None


class Cuboid(Polyhedron):
    """A 3D cuboid geometry"""
    def __init__(self, frame: 'Frame', lx: _fnp, ly: _fnp, lz: _fnp):
        """Initialises a cuboid geometry of the given size.

        :param frame: The cuboid's frame. This frame's origin defines the centre of the cuboid. The x, y and z faces of
        the cube are each centred and perpendicular to the x, y and z axes of the frame respectively.
        :param lx: Cuboid length along its x-axis
        :param ly: Cuboid length along its y-axis
        :param lz: Cuboid length along its z-axis
        """
        self.lx = lx
        self.ly = ly
        self.lz = lz
        super().__init__(frame)

    @property
    def centre(self) -> 'Vec3':
        """The geometric centre of the cuboid"""
        return self._frame.origin

    @property
    def faces(self) -> List[Rectangle]:
        """The px, nx, py, ny, pz, nz faces of the cuboid"""
        frame = self._frame.translated(0.5 * self.lz * self._frame.w)
        fzp = Rectangle(frame, self.lx, self.ly)
        frame = frame.rotated(self.centre, self._frame.u, np.radians(180))
        fzn = Rectangle(frame, self.lx, self.ly)
        frame = self._frame.translated(0.5 * self.lx * self._frame.u).rotatedInPlace(self._frame.v, np.radians(90))
        fxp = Rectangle(frame, self.lz, self.ly)
        frame = self._frame.translated(-0.5 * self.lx * self._frame.u).rotatedInPlace(self._frame.v, np.radians(270))
        fxn = Rectangle(frame, self.lz, self.ly)
        frame = self._frame.translated(0.5 * self.ly * self._frame.v).rotatedInPlace(self._frame.u, np.radians(270))
        fyp = Rectangle(frame, self.lx, self.lz)
        frame = self._frame.translated(-0.5 * self.ly * self._frame.v).rotatedInPlace(self._frame.u, np.radians(90))
        fyn = Rectangle(frame, self.lx, self.lz)
        return [fxp, fxn, fyp, fyn, fzp, fzn]

    @property
    def isNumpyType(self) -> bool:
        nump = np.ndarray
        return type(self.lx) is nump or type(self.ly) is nump or type(self.lz) is nump or super().isNumpyType

    def normal(self, point: 'Vec3') -> 'Vec3':
        local = self._frame.fromWorld(point)
        dpx = abs(local.x - 0.5 * self.lx)
        dnx = abs(local.x + 0.5 * self.lx)
        dpy = abs(local.y - 0.5 * self.ly)
        dny = abs(local.y + 0.5 * self.ly)
        dpz = abs(local.z - 0.5 * self.lz)
        dnz = abs(local.z + 0.5 * self.lz)
        distances = dpx, dnx, dpy, dny, dpz, dnz
        if point.isNumpyType or self.isNumpyType:
            shortest = dpx
            for distance in distances[1:]:
                shortest = np.minimum(shortest, distance)
            vx = 1, -1, 0, 0, 0, 0
            vy = 0, 0, 1, -1, 0, 0
            vz = 0, 0, 0, 0, 1, -1
            resultx = np.full_like(dpx, np.nan, dtype=float)
            resulty = np.full_like(dpx, np.nan, dtype=float)
            resultz = np.full_like(dpx, np.nan, dtype=float)
            for index, distance in enumerate(distances):
                matching = distance == shortest
                resultx[matching] = vx[index]
                resulty[matching] = vy[index]
                resultz[matching] = vz[index]
            return self.frame.toWorld(Vec3((resultx, resulty, resultz)))
        else:
            shortest = min(distances)
            for n, d in enumerate(distances):
                if d == shortest:
                    return self.faces[n].n
            return Vec3.zero()

    def uv(self, point: Vec3) -> Optional[Tuple[_fnp, _fnp]]:
        i = self.faceFromSurfacePoint(point)
        plocal = self.frame.fromWorld(point)
        x0s = (1.5 * self.lz + self.lx, 0.5 * self.lz, self.lz + 0.5 * self.lx, self.lz + 0.5 * self.lx,
               self.lz + 0.5 * self.lx, 2 * self.lz + 1.5 * self.lx)
        y0s = (self.lz + 0.5 * self.ly, self.lz + 0.5 * self.ly, 0.5 * self.lz, 1.5 * self.lz + self.ly,
               self.lz + 0.5 * self.ly, self.lz + 0.5 * self.ly)
        dxs = (-plocal.z, plocal.z, plocal.x, plocal.x, plocal.x, -plocal.x)
        dys = (-plocal.y, -plocal.y, plocal.z, -plocal.z, -plocal.y, -plocal.y)
        if point.isNumpyType or type(i) is np.ndarray:
            x = np.full_like(i, np.nan, dtype=float)
            y = np.full_like(i, np.nan, dtype=float)
            for j in range(6):
                cond = i.astype(int) == j
                x[cond] = x0s[j] + dxs[j][cond]
                y[cond] = y0s[j] + dys[j][cond]
        else:
            x = x0s[i] + dxs[i]
            y = y0s[i] + dys[i]
        u = 0.5 * x / (self.lx + self.lz)
        v = y / (2 * self.lz + self.ly)
        return u, v

    def pointFromUV(self, u: _fnp, v: _fnp) -> Optional[Vec3]:
        a = u * 2 * (self.lx + self.lz)
        b = v * (2 * self.lz + self.ly)
        a0s = (1.5 * self.lz + self.lx, 0.5 * self.lz, self.lz + 0.5 * self.lx, self.lz + 0.5 * self.lx,
               self.lz + 0.5 * self.lx, 2 * self.lz + 1.5 * self.lx)
        b0s = (self.lz + 0.5 * self.ly, self.lz + 0.5 * self.ly, 0.5 * self.lz, 1.5 * self.lz + self.ly,
               self.lz + 0.5 * self.ly, self.lz + 0.5 * self.ly)
        aAxs = (-Vec3.k(), Vec3.k(), Vec3.i(), Vec3.i(), Vec3.i(), -Vec3.i())
        bAxs = (-Vec3.j(), -Vec3.j(), Vec3.k(), -Vec3.k(), -Vec3.j(), -Vec3.j())
        faceCentres = (0.5 * self.lx * Vec3.i(), -0.5 * self.lx * Vec3.i(), 0.5 * self.ly * Vec3.j(),
                       -0.5 * self.ly * Vec3.j(), 0.5 * self.lz * Vec3.k(), -0.5 * self.lz * Vec3.k())

        i = self.faceFromUV(u, v)
        if type(i) is np.ndarray:
            retX = np.full_like(i, np.nan, dtype=float)
            retY = np.full_like(i, np.nan, dtype=float)
            retZ = np.full_like(i, np.nan, dtype=float)
            for j in range(6):
                cond = i == j
                da = a - a0s[j]
                db = b - b0s[j]
                retX[cond] = da[cond] * aAxs[j].x + db[cond] * bAxs[j].x + faceCentres[j].x
                retY[cond] = da[cond] * aAxs[j].y + db[cond] * bAxs[j].y + faceCentres[j].y
                retZ[cond] = da[cond] * aAxs[j].z + db[cond] * bAxs[j].z + faceCentres[j].z
            plocal = Vec3((retX, retY, retZ))
        else:
            plocal = (a - a0s[i]) * aAxs[i] + (b - b0s[i]) * bAxs[i] + faceCentres[i]
        return self.frame.toWorld(plocal)

    def intersect(self, ray: 'Ray', epsilon=1e-7, max_dist: float = 1e15, shift: Vec3 = None) -> Dict[str, np.ndarray]:
        ret = {}
        if shift is not None:
            self.frame = self.frame.translated(shift)
        raylocal = ray.transformed(new=self._frame)
        if self.isNumpyType or ray.isNumpyType:
            dinv = Vec3((1 / raylocal.d.x, 1 / raylocal.d.y, 1 / raylocal.d.z))
            tx1 = ((-0.5 * self.lx) - raylocal.origin.x) * dinv.x
            tx2 = ((0.5 * self.lx) - raylocal.origin.x) * dinv.x
            tmin = np.minimum(tx1, tx2)
            tmax = np.maximum(tx1, tx2)
            ty1 = ((-0.5 * self.ly) - raylocal.origin.y) * dinv.y
            ty2 = ((0.5 * self.ly) - raylocal.origin.y) * dinv.y
            tmin = np.maximum(tmin, np.minimum(ty1, ty2))
            tmax = np.minimum(tmax, np.maximum(ty1, ty2))
            tz1 = ((-0.5 * self.lz) - raylocal.origin.z) * dinv.z
            tz2 = ((0.5 * self.lz) - raylocal.origin.z) * dinv.z
            tmin = np.maximum(tmin, np.minimum(tz1, tz2))
            tmax = np.minimum(tmax, np.maximum(tz1, tz2))
            miss = tmin > tmax
            inside = ((tmin < 0) * (tmax > 0)) + ((tmin > 0) * (tmax < 0))
            behind = (tmin < 0) * (tmax < 0)
            valid = ~miss * ~inside * ~behind
            depth = np.full_like(tmin, np.nan, dtype=float)
            depth[valid] = tmin[valid]
            inRange = depth <= max_dist
            depth[~inRange] = np.nan
            ret['t_hit'] = depth
            point = ray.point(depth)
            n = self.normal(point)
            ret['primitive_normals'] = n.asNumpyArray
            ret['primitive_uvs'] = Vec2(self.uv(point)).asNumpyArray
        else:
            if raylocal.d.x == 0:
                raylocal.d.x = epsilon
            if raylocal.d.y == 0:
                raylocal.d.y = epsilon
            if raylocal.d.z == 0:
                raylocal.d.z = epsilon
            dinv = Vec3((1 / raylocal.d.x, 1 / raylocal.d.y, 1 / raylocal.d.z))
            tx1 = ((-0.5 * self.lx) - raylocal.origin.x) * dinv.x
            tx2 = ((0.5 * self.lx) - raylocal.origin.x) * dinv.x
            tmin = min(tx1, tx2)
            tmax = max(tx1, tx2)
            ty1 = ((-0.5 * self.ly) - raylocal.origin.y) * dinv.y
            ty2 = ((0.5 * self.ly) - raylocal.origin.y) * dinv.y
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))
            tz1 = ((-0.5 * self.lz) - raylocal.origin.z) * dinv.z
            tz2 = ((0.5 * self.lz) - raylocal.origin.z) * dinv.z
            tmin = max(tmin, min(tz1, tz2))
            tmax = min(tmax, max(tz1, tz2))
            miss = tmin > tmax
            inside = ((tmin < 0) * (tmax > 0)) + ((tmin > 0) * (tmax < 0))
            behind = (tmin < 0) * (tmax < 0)
            outRange = tmin > max_dist
            if not miss and not inside and not behind and not outRange:
                ret['t_hit'] = tmin
                point = ray.point(tmin)
                ret['primitive_normals'] = self.normal(point).asNumpyArray
                ret['primitive_uvs'] = Vec2(self.uv(point)).asNumpyArray
        if shift is not None:
            self.frame = self.frame.translated(-shift)

        return ret

    def faceFromSurfacePoint(self, point: 'Vec3') -> _inp:
        """Returns the index of the face of this cuboid that is closest to the given point (where indices 0-5 correspond
        to px, nx, py, ny, pz, nz faces respectively)

        :param point: the point, in world coordinates (this should be on or very close to a face, otherwise the result
                    will be inaccurate)
        :return: the index of the closest face
        """
        local = self._frame.fromWorld(point)
        dpx = abs(local.x - 0.5 * self.lx)
        dnx = abs(local.x + 0.5 * self.lx)
        dpy = abs(local.y - 0.5 * self.ly)
        dny = abs(local.y + 0.5 * self.ly)
        dpz = abs(local.z - 0.5 * self.lz)
        dnz = abs(local.z + 0.5 * self.lz)
        distances = dpx, dnx, dpy, dny, dpz, dnz
        if point.isNumpyType or self.isNumpyType:
            shortest = dpx
            for distance in distances[1:]:
                shortest = np.minimum(shortest, distance)
            result = np.zeros_like(dpx, dtype=int)
            for n, distance in enumerate(distances[1:]):
                matching = distance == shortest
                result[matching] = n + 1
            return result
        else:
            shortest = min(distances)
            for n, d in enumerate(distances):
                if d == shortest:
                    return n
            return 0

    def faceFromUV(self, u: _fnp, v: _fnp) -> Optional[_inp]:
        """ For the given uv cuboid texture coordinate, this function returns the corresponding face of the cuboid
        as an index (where indices 0-5 correspond to px, nx, py, ny, pz, nz faces respectively)

        :param u: u texture coordinate
        :param v: v texture coordinate
        :return: cuboid face index
        """
        a = u * 2 * (self.lx + self.lz)
        b = v * (2 * self.lz + self.ly)
        pxCond = (a >= self.lz + self.lx) * (a < 2 * self.lz + self.lx) * (b >= self.lz) * (b < self.lz + self.ly)
        nxCond = (a >= 0) * (a < self.lz) * (b >= self.lz) * (b < self.lz + self.ly)
        pyCond = (a >= self.lz) * (a < self.lz + self.lx) * (b >= 0) * (b < self.lz)
        nyCond = (a >= self.lz) * (a < self.lz + self.lx) * (b >= self.lz + self.ly) * (v < 1)
        pzCond = (a >= self.lz) * (a < self.lz + self.lx) * (b >= self.lz) * (b < self.lz + self.ly)
        nzCond = (a >= 2 * self.lz + self.lx) * (u < 1) * (b >= self.lz) * (b < self.lz + self.ly)
        conds = (pxCond, nxCond, pyCond, nyCond, pzCond, nzCond)
        if type(a) is np.ndarray:
            ret = np.full_like(a, np.nan, dtype=float)
            for i, cond in enumerate(conds):
                ret[cond] = i
            return ret
        else:
            for i, cond in enumerate(conds):
                if cond:
                    return i
            return None


class Spheroid(Geometry):
    """A spheroid geometry"""
    def __init__(self, frame: 'Frame', rx: _fnp, ry: _fnp, rz: _fnp):
        """Initialises a spheroid geometry

        :param frame: The spheroid's frame. This frame's origin defines the centre of the spheroid.
        :param rx: Spheroid radius [m] measured along its x-axis
        :param ry: Spheroid radius [m] measured along its y-axis
        :param rz: Spheroid radius [m] measured along its z-axis
        """
        self.rx = rx
        self.ry = ry
        self.rz = rz
        super().__init__(frame)

    @property
    def centre(self) -> 'Vec3':
        """The geometric centre of the spheroid"""
        return self._frame.origin

    @property
    def isNumpyType(self) -> bool:
        nump = np.ndarray
        return type(self.rx) is nump or type(self.ry) is nump or type(self.rz) is nump or super().isNumpyType

    def normal(self, point: 'Vec3') -> 'Vec3':
        pointlocal = self._frame.fromWorld(point)
        nlocal = Vec3((pointlocal.x * self.rx ** -2, pointlocal.y * self.ry ** -2, pointlocal.z * self.rz ** -2)).norm
        return self.frame.toWorld(pointlocal + nlocal) - point

    def uv(self, point: Vec3) -> Optional[Tuple[_fnp, _fnp]]:
        _, az, pol = self.frame.fromWorldToSpherical(point)
        az = az % (2 * np.pi)
        u = 0.5 * az / np.pi
        v = pol / np.pi
        return u, v

    def pointFromUV(self, u: _fnp, v: _fnp) -> Optional[Vec3]:
        az = 2 * np.pi * u
        pol = np.pi * v
        r = self.radius(az, pol)
        return self.frame.fromSphericalToWorld(r, az, pol)

    def intersect(self, ray: 'Ray', epsilon=1e-7, max_dist: float = 1e15, shift: Vec3 = None) -> Dict[str, np.ndarray]:
        ret = {}
        if shift is not None:
            self.frame = self.frame.translated(shift)
        raylocal = ray.transformed(new=self._frame)
        fracx = self.rx ** -2
        fracy = self.ry ** -2
        fracz = self.rz ** -2
        o = raylocal.origin
        d = raylocal.d
        a = d.x ** 2 * fracx + d.y ** 2 * fracy + d.z ** 2 * fracz
        b = 2 * (o.x * d.x * fracx + o.y * d.y * fracy + o.z * d.z * fracz)
        c = o.x ** 2 * fracx + o.y ** 2 * fracy + o.z ** 2 * fracz - 1
        root1, root2 = math2.quadSolve(a, b, c)
        if ray.isNumpyType or self.isNumpyType:
            infdist = 2 * max_dist
            root1[np.isnan(root1) + (root1 <= 0)] = infdist
            root2[np.isnan(root2) + (root2 <= 0)] = infdist
            depth = np.minimum(root1, root2)
            depth[depth > max_dist] = np.nan
            ret['t_hit'] = depth
            point = ray.point(depth)  # intersection point in world coordinates
            ret['primitive_normals'] = self.normal(point).asNumpyArray
            _, az, pol = self.frame.fromWorldToSpherical(point)
            az = az % (2 * np.pi)
            u = 0.5 * az / np.pi
            v = pol / np.pi
            uv = Vec2((u, v))
            ret['primitive_uvs'] = uv.asNumpyArray
        else:
            if root1 <= 0:
                root1 = None
            if root2 <= 0:
                root2 = None
            if root1 is None and root2 is None:
                if shift is not None:
                    self.frame = self.frame.translated(-shift)
                return {}
            elif root1 is None:
                depth = root2
            elif root2 is None:
                depth = root1
            else:
                depth = min(root1, root2)
            if depth > max_dist:
                if shift is not None:
                    self.frame = self.frame.translated(-shift)
                return {}
            ret['t_hit'] = depth
            point = ray.point(depth)
            ret['primitive_normals'] = self.normal(point).asNumpyArray
            _, az, pol = self.frame.fromWorldToSpherical(point)
            az = az % (2 * np.pi)
            u = 0.5 * az / np.pi
            v = pol / np.pi
            uv = Vec2((u, v))
            ret['primitive_uvs'] = uv.asNumpyArray

        if shift is not None:
            self.frame = self.frame.translated(-shift)

        return ret

    def radius(self, az: _fnp, pol: _fnp) -> _fnp:
        """ Returns the radius of this spheroid at the given azimuth, polar coordinate on its surface

        :param az: azimuth [radians] of point at which radius should be calculated
        :param pol: polar angle [radians] of point at which radius should be calculated
        :return: radius (surface distance from centre) [m] at given point
        """
        _sp = np.sin(pol)
        denom = ((np.cos(az) ** 2) * (_sp ** 2)) / (self.rx ** 2)
        denom = denom + ((np.sin(az) ** 2) * (_sp ** 2)) / (self.ry ** 2)
        del _sp
        denom = denom + (np.cos(pol) ** 2) / (self.rz ** 2)
        return (1 / denom) ** 0.5
