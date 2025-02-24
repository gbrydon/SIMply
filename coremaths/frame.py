"""
Module providing functionality for describing and converting between coordinate frames.
"""
import math
import numpy as np
import scipy
from coremaths.vector import Vec3, Mat3, Quaternion
from typing import List, Tuple, Union

_fnp = Union[float, np.ndarray]
_3dPoints = Union[np.ndarray, Vec3, List[Vec3]]


class Frame:
    """A 3D cartesian right-handed coordinate frame.

    The vectors describing this coordinate frame can have numpy arrays as their components.
    If this is the case for more than one component/vector, all numpy arrays must have the same shape.
    """

    @staticmethod
    def _convertBetweenFrames(point: Vec3, frame_a: 'Frame', frame_b: 'Frame') -> Vec3:
        """ Returns the result of converting a point from one right-handed 3D cartesian frame to another.

        :param point: the point's cartesian coordinates in frame A.
        :param frame_a: frame A.
        :param frame_b: frame B.
        :return: the point's cartesian coordinates in frame B.
        """
        return frame_b.matrixInverse * (frame_a.origin - frame_b.origin + (frame_a.matrix * point))

    @staticmethod
    def _rigidTransform(frame_a: 'Frame', frame_b: 'Frame') -> Tuple[Mat3, Vec3]:
        """ Returns the 3x3 rotation matrix R and translation vector t that describe the rigid transform from the
        given 3D cartesian coordinate frame_a to the given 3D cartesian coordinate frame_b.

        The returned R and t describe the rigid transform of a point from frame_a to frame_b
        by pb = Rpa + t, where pa is the point's coordinate in frame_a and pb is its coordinate in frame_b.

        :param frame_a: frame a.
        :param frame_b: frame b.
        :return: tuple containing the rotation matrix and translation vector giving the rigid transform from a to b.
        """
        R = frame_b.matrixInverse * frame_a.matrix
        t = frame_b.matrixInverse * (frame_a.origin - frame_b.origin)
        return R, t

    @staticmethod
    def rigidTransformFromPoints(points_a: _3dPoints, points_b: _3dPoints) -> Tuple[Mat3, Vec3]:
        """ Given two sets of 3D coordinates that represent the same points but in two different coordinate frames
        (a and b), this function finds the optimal (minimal least-squares) rigid transform (rotation and translation)
        that converts points from frame a coordinates to frame b coordinates.

        A minimum of 3 points is required.

        points_a and points_b must be of the same type as each other, and can either be nx3 numpy arrays representing n
        points (e.g. [[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]]), numpy-type Vec3 objects, or lists of non-numpy-type
        Vec3 objects. points_a and points_b are the coordinates of the same points, but in different coordinate frames,
        so must store the same number of coordinates as each other.

        The rigid transform is returned as a rotation matrix R and translation vector t which together transform points
        in frame a (pa) to points in frame b (pb) by pb=Rpa+t.

        Rigid transforms do not involve scaling (i.e. distances between points remain the same). If scaling is also
        required (i.e. distances between points are different in points_a than in points_b), use
        Frame.similarityTransformFromPoints(...) instead.


        :param points_a: the points' coordinates in frame a.
        :param points_b: the points' coordinates in frame b.
        :return: the rigid transform - tuple containing rotation matrix R and translation vector t.
        """
        if type(points_a) is not type(points_b):
            message = ("points_a and points_b must either both be numpy arrays, both be Vec3, or both be a list of "
                       "Vec3s, but have types of {0} and {1} respectively".format(type(points_a), type(points_b)))
            raise TypeError(message)
        if type(points_a) is Vec3:
            shape_a = points_a.numpyShape
            shape_b = points_b.numpyShape
            if shape_a != shape_b or len(shape_a) != 1:
                message = ("Vec3 objects points_a and points_b must have matching numpy shape of n, but have shapes "
                           "{0} and {1} respectively".format(shape_a, shape_b))
                raise ValueError(message)
        elif type(points_a) is list:
            if type(points_a[0]) is Vec3:
                la, lb = len(points_a), len(points_b)
                if la != lb:
                    message = ("points_a and points_b lists of Vec3s must have same length, but have lengths {0} and"
                               " {1} respectively".format(la, lb))
                    raise ValueError(message)
            else:
                message = ("points_a and points_b must either both be numpy arrays, both be Vec3, or both be a list of "
                           "Vec3s, but both are lists of {}".format(type(points_a[0])))
                raise TypeError(message)
        elif type(points_a) is np.ndarray:
            shape_a = points_a.shape
            shape_b = points_b.shape
            if shape_a != shape_b or len(shape_a) != 2 or shape_a[1] != 3:
                message = ("numpy arrays for points_a and points_b must have matching shape of nx3, but have shapes "
                           "{0} and {1} respectively.".format(shape_a, shape_b))
                raise ValueError(message)
        else:
            message = ("points_a and points_b must either both be numpy arrays, both be Vec3, or both be a list of "
                       "Vec3s, but both have types of {}".format(type(points_a)))
            raise TypeError(message)

        if type(points_a) is Vec3:
            pA = points_a.asNumpyArray.T
            pB = points_b.asNumpyArray.T
        elif type(points_a) is np.ndarray:
            pA = points_a.T
            pB = points_b.T
        else:
            pA = np.empty((3, len(points_a)))
            pB = np.empty((3, len(points_a)))
            for n, vecA in enumerate(points_a):
                vecB = points_b[n]
                pA[:, n] = vecA.tuple
                pB[:, n] = vecB.tuple

        centA = np.mean(pA, axis=1)
        centA = centA[:, None]  # change centA from numpy array of shape (3,) to shape (3, 1)
        centB = np.mean(pB, axis=1)
        centB = centB[:, None]  # change centB from numpy array of shape (3,) to shape (3, 1)

        zeroedA = pA - centA
        zeroedB = pB - centB

        H = zeroedA @ np.transpose(zeroedB)
        u, s, vTr = np.linalg.svd(H)
        R = vTr.T @ u.T

        # handle special reflection case
        if np.linalg.det(R) < 0:
            # special reflection case has occurred, must be corrected for
            vTr[2, :] *= -1
            R = vTr.T @ u.T

        t = -R @ centA + centB

        return Mat3(R[0], R[1], R[2]), Vec3(t)

    @staticmethod
    def similarityTransformFromPoints(points_a: _3dPoints, points_b: _3dPoints) -> Tuple[Mat3, Vec3, float]:
        """ Given two sets of 3D coordinates that represent the same points but in different coordinate frames
        (a and b) with different scales, this function finds the optimal (minimal least-squares) similarity transform
        (rotation, translation and scaling) that converts points from frame a coordinates to frame b coordinates.

        A minimum of 3 points is required.

        points_a and points_b must be of the same type as each other, and can either be nx3 numpy arrays representing n
        points (e.g. [[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]]), numpy-type Vec3 objects, or lists of non-numpy-type
        Vec3 objects. points_a and points_b are the coordinates of the same points, but in different coordinate frames,
        so must store the same number of coordinates as each other.

        The similarity transform is returned as a rotation matrix R, translation vector t and scale scalar s,
        which together transform points in frame a (pa) to points in frame b (pb) by pb=s*R*pa+t.

        If there is no scale difference between the points' representations in points_a and points_b, use
        Frame.rigidTransformFromPoints() instead.

        :param points_a: the points' coordinates in frame a.
        :param points_b: the points' coordinates in frame b.
        :return: the rigid transform - tuple containing rotation matrix R, translation vector t and scale value s.
        """
        if type(points_a) is not type(points_b):
            message = ("points_a and points_b must either both be numpy arrays, both be Vec3, or both be a list of "
                       "Vec3s, but have types of {0} and {1} respectively".format(type(points_a), type(points_b)))
            raise TypeError(message)
        if type(points_a) is Vec3:
            shape_a = points_a.numpyShape
            shape_b = points_b.numpyShape
            if shape_a != shape_b or len(shape_a) != 1:
                message = ("Vec3 objects points_a and points_b must have matching numpy shape of n, but have shapes "
                           "{0} and {1} respectively".format(shape_a, shape_b))
                raise ValueError(message)
        elif type(points_a) is list:
            if type(points_a[0]) is Vec3:
                la, lb = len(points_a), len(points_b)
                if la != lb:
                    message = ("points_a and points_b lists of Vec3s must have same length, but have lengths {0} and"
                               " {1} respectively".format(la, lb))
                    raise ValueError(message)
            else:
                message = ("points_a and points_b must either both be numpy arrays, both be Vec3, or both be a list of "
                           "Vec3s, but both are lists of {}".format(type(points_a[0])))
                raise TypeError(message)
        elif type(points_a) is np.ndarray:
            shape_a = points_a.shape
            shape_b = points_b.shape
            if shape_a != shape_b or len(shape_a) != 2 or shape_a[1] != 3:
                message = ("numpy arrays for points_a and points_b must have matching shape of nx3, but have shapes "
                           "{0} and {1} respectively.".format(shape_a, shape_b))
                raise ValueError(message)
        else:
            message = ("points_a and points_b must either both be numpy arrays, both be Vec3, or both be a list of "
                       "Vec3s, but both have types of {}".format(type(points_a)))
            raise TypeError(message)

        if type(points_a) is Vec3:
            pA = points_a.asNumpyArray
            pB = points_b.asNumpyArray
        elif type(points_a) is np.ndarray:
            pA = points_a
            pB = points_b
        else:
            pA = np.empty((len(points_a), 3))
            pB = np.empty((len(points_a), 3))
            for n, vec3A in enumerate(points_a):
                vec3B = points_b[n]
                pA[n, :] = vec3A.tuple
                pB[n, :] = vec3B.tuple

        vA = pA[1:, :] - pA[:-1, :]
        vB = pB[1:, :] - pB[:-1, :]
        lA = Vec3((vA[:, 0], vA[:, 1], vA[:, 2])).length
        lB = Vec3((vB[:, 0], vB[:, 1], vB[:, 2])).length

        def objective(scale):
            dl = scale * lA - lB
            return np.sum(dl ** 2)

        res = scipy.optimize.minimize(objective, np.mean(lB / lA))
        s = res['x']

        return Frame.rigidTransformFromPoints(s * pA, pB) + (s,)

    @staticmethod
    def changeBasis(vector: 'Vec3', frame_a: 'Frame', frame_b: 'Frame') -> Vec3:
        """ Returns the result of changing a vector's basis from frame_a's axes to frame_b's.

        :param vector: the vector in frame_a's basis
        :param frame_a: the frame from which the vector's basis is being changed
        :param frame_b: the frame to which the vector's basis is being changed
        :return: the vector in frame_b's basis
        """
        return frame_b.matrixInverse * (frame_a.matrix * vector)

    @staticmethod
    def toSpherical(point: 'Vec3') -> Tuple[_fnp, _fnp, _fnp]:
        """ Returns the spherical coordinate of a given cartesian coordinate (both with respect to the same frame).

        :param point: the cartesian coordinate.
        :return: the spherical polar coordinate as (radius, azimuth, polar) tuple.
        """
        r = point.length
        try:
            polar = math.acos(point.z / r)
            azimuth, _ = Vec3.i().anticlockAngleWith(point.ontoXY, Vec3.k())
            return r, azimuth, polar
        except TypeError:
            polar = np.arccos(point.z / r)
            azimuth, _ = Vec3.i().anticlockAngleWith(point.ontoXY, Vec3.k())
            return r, azimuth, polar

    @staticmethod
    def fromSpherical(radius: _fnp, azimuth: _fnp, polar: _fnp) -> Vec3:
        """Returns the cartesian coordinate of given spherical coordinate (both with respect to the same frame).

        :param radius: spherical radius of point.
        :param azimuth: spherical azimuth angle (in radians) of point, measured from x-axis around z-axis.
        :param polar: spherical polar angle (in radians) of point, measured from z-axis.
        :return: the cartesian coordinate.
        """
        try:
            x = radius * math.sin(polar) * math.cos(azimuth)
            y = radius * math.sin(polar) * math.sin(azimuth)
            z = radius * math.cos(polar)
        except TypeError:
            x = radius * np.sin(polar) * np.cos(azimuth)
            y = radius * np.sin(polar) * np.sin(azimuth)
            z = radius * np.cos(polar)
        return Vec3((x, y, z))

    @staticmethod
    def toEquatorial(point: 'Vec3') -> Tuple[_fnp, _fnp]:
        """Returns the astronomical equatorial coordinate of the given cartesian coordinate in the J2000 frame as seen
        from the origin

        :param point: the cartesian coordinate
        :return: the RA and dec of the point as seen from frame's origin [radians]
        """
        _, az, pol = Frame.toSpherical(point)
        dec = 0.5 * math.pi - pol
        return az, dec

    @staticmethod
    def fromEquatorial(ra: _fnp, dec: _fnp, radius=1e15) -> 'Vec3':
        """Returns the cartesian coordinates of an object in the J2000 frame with a given RA, dec and radius from origin

        :param ra: the object's RA [radians]
        :param dec: the object's declination [radians]
        :param radius: the radius of the object (1e12 km by default) [m]
        :return: the cartesian coordinate
        """
        pol = 0.5 * math.pi - dec
        return Frame.fromSpherical(radius, ra, pol)

    @staticmethod
    def toLongLat(point: 'Vec3') -> Tuple[_fnp, _fnp]:
        """ Returns the longitude and latitude coordinates of a point given by a 3D vector in the same coordinate frame.
        Longitude is measured from the coordinate frame's x-axis, anticlockwise around the z-axis. Latitude is measured
        from the frame's xy plane, and is +ve for points with +ve z and -ve for points with -ve z.

        :param point: cartesian coordinate of point
        :return: longitude, latitude tuple [radians]
        """
        _, long, pol = Frame.toSpherical(point)
        lat = 0.5 * math.pi - pol
        return long, lat

    @staticmethod
    def _frameToFrameQuaternion(frame1: 'Frame', frame2: 'Frame') -> 'Quaternion':
        """ Returns the quaternion required to rotate one frame onto the same orientation as another

        :param frame1: the frame to rotate
        :param frame2: the frame onto which the first frame will be rotated
        :return: the rotation quaternion
        """
        ang1, ax1 = frame1.u.anticlockAngleWith(frame2.u)
        intermediate = frame1.rotatedInPlace(ax1, ang1)
        ang2, _ = intermediate.v.anticlockAngleWith(frame2.v, frame2.u)
        q1 = Quaternion.fromRotationParams(ax1, ang1)
        q2 = Quaternion.fromRotationParams(frame2.u, ang2)
        return q2 * q1

    def __init__(self, u: 'Vec3', v: 'Vec3', w: 'Vec3', origin: 'Vec3' = Vec3.zero()):
        """ Initialises a new frame from the given principal axes.

        Any of the vectors defining this frame can be of numpy-type (see Vec3 documentation). If multiple are of
        numpy-type, they must have the same numpy shape.

        :param u: The direction vector of the frame's 'i' axis.
        :param v: The direction vector of the frame's 'j' axis.
        :param w: The direction vector of the frame's 'k' axis.
        :param origin: The frame's origin.
        """
        self._origin = origin
        self._u = u.norm
        self._v = v.norm
        self._w = w.norm
        top = (u[0], v[0], w[0])
        middle = (u[1], v[1], w[1])
        bottom = (u[2], v[2], w[2])
        self._matrix = Mat3(top, middle, bottom)
        self._matrixInverse = self.matrix.transpose

    @classmethod
    def world(cls) -> 'Frame':
        """The world coordinate frame (origin = [0, 0, 0]; u=[1, 0, 0]; v=[0, 1, 0]; w=[0, 0, 1])."""
        return cls(Vec3.i(), Vec3.j(), Vec3.k(), Vec3.zero())

    @classmethod
    def worldAligned(cls, origin: 'Vec3') -> 'Frame':
        """ Coordinate frame with the given origin location and aligned with the world frame.

        :param origin: the location of the frame's origin
        :return: the frame
        """
        return cls(Vec3.i(), Vec3.j(), Vec3.k(), origin)

    @classmethod
    def withU(cls, u: 'Vec3', origin=Vec3.zero(), v: 'Vec3' = None, w: 'Vec3' = None) -> 'Frame':
        """ A frame with given u axis.

        :param u: Normalised vector giving the direction of the frame's u axis.
        :param origin: The frame's origin ([0, 0, 0] by default).
        :param v: Is None by default. However, if a vector is passed for v (must be normalised), the returned frame will
                    be oriented to align its v axis as closely as possible with this vector. If None is passed,
                    orientation of v axis will be arbitrary, or dictated by vector passed for 'w' if any.
        :param w: Is None by default. However, if a vector is passed for w (must be normalised), the returned frame will
                    be oriented to align its w axis as closely as possible with this vector (unless a vector for v is
                    also passed, in which case w argument is ignored). If None is passed, orientation of w axis will be
                    arbitrary, or dictated by vector passed for 'v' if any.
        :return: The frame with desired u axis.
        """
        if v is not None:
            w = u.cross(v).norm
            v = w.cross(u).norm
            return cls(u, v, w, origin)
        if w is not None:
            v = w.cross(u).norm
            w = u.cross(v).norm
            return cls(u, v, w, origin)
        v = Vec3.vectorPerpendicularTo(u)
        w = u.cross(v).norm
        return cls(u, v, w, origin)

    @classmethod
    def withV(cls, v: 'Vec3', origin=Vec3.zero(), u: 'Vec3' = None, w: 'Vec3' = None) -> 'Frame':
        """ A frame with given v axis.

        :param v: Normalised vector giving the direction of the frame's v axis
        :param origin: The frame's origin.
        :param u: Is None by default. However, if a vector is passed for u (must be normalised), the returned frame will
                    be oriented to align its u axis as closely as possible with this vector. If None is passed,
                    orientation of u axis will be arbitrary, or dictated by vector passed for 'w' if any.
        :param w: Is None by default. However, if a vector is passed for w (must be normalised), the returned frame will
                    be oriented to align its w axis as closely as possible with this vector (unless a vector for u is
                    also passed, in which case w argument is ignored). If None is passed, orientation of w axis will be
                    arbitrary, or dictated by vector passed for 'u' if any.
        :return: The frame with desired v axis
        """
        if u is not None:
            w = u.cross(v).norm
            u = v.cross(w).norm
            return cls(u, v, w, origin)
        if w is not None:
            u = v.cross(w).norm
            w = u.cross(v).norm
            return cls(u, v, w, origin)
        u = Vec3.vectorPerpendicularTo(v)
        w = u.cross(v).norm
        return cls(u, v, w, origin)

    @classmethod
    def withW(cls, w: 'Vec3', origin=Vec3.zero(), u: 'Vec3' = None, v: 'Vec3' = None) -> 'Frame':
        """ A frame with given w axis and arbitrary u and v axes.

        :param w: Normalised vector giving the direction of the frame's w axis
        :param origin: The frame's origin
        :param u: Is None by default. However, if a vector is passed for u (must be normalised), the returned frame will
                    be oriented to align its u axis as closely as possible with this vector. If None is passed,
                    orientation of u axis will be arbitrary, or dictated by vector passed for 'v' if any.
        :param v: Is None by default. However, if a vector is passed for v (must be normalised), the returned frame will
                    be oriented to align its v axis as closely as possible with this vector (unless a vector for u is
                    also passed, in which case v argument is ignored). If None is passed, orientation of v axis will be
                    arbitrary, or dictated by vector passed for 'u' if any.
        :return: The frame with desired w axis
        """
        if u is not None:
            v = w.cross(u).norm
            u = v.cross(w).norm
            return cls(u, v, w, origin)
        if v is not None:
            u = v.cross(w).norm
            v = w.cross(u).norm
            return cls(u, v, w, origin)
        u = Vec3.vectorPerpendicularTo(w)
        v = w.cross(u).norm
        return cls(u, v, w, origin)

    @property
    def isNumpyType(self) -> 'bool':
        """Whether any of the vectors defining this frame are of numpy type (see Vec3 documentation)."""
        return self._origin.isNumpyType or self._u.isNumpyType or self._v.isNumpyType or self._w.isNumpyType

    def fromWorld(self, point: 'Vec3') -> 'Vec3':
        """Returns the result of converting a coordinate from the world frame to this frame.
        :param point: the point to convert to this frame
        :return: the converted point
        """
        return self._matrixInverse * (point - self._origin)

    def toWorld(self, point: 'Vec3') -> 'Vec3':
        """ Returns the result of converting a coordinate from this frame to the world frame.

        :param point: the point to convert to the world frame
        :return: the converted point
        """
        return self._origin + self._matrix * point

    def toFrame(self, point: 'Vec3', new_frame: 'Frame') -> 'Vec3':
        """Returns the result of converting a coordinate from this frame to another.

        :param point: The point to convert
        :param new_frame: The frame to convert the point to
        :return: The converted point
        """
        return Frame._convertBetweenFrames(point, self, new_frame)

    def fromFrame(self, point: 'Vec3', origin_frame: 'Frame') -> 'Vec3':
        """Returns the result of converting a coordinate to this frame from another.

        :param point: The point to convert
        :param origin_frame: The frame to convert the point from
        :return: The converted point
        """
        return Frame._convertBetweenFrames(point, origin_frame, self)

    def rigidTransformTo(self, frame: 'Frame') -> Tuple['Mat3', 'Vec3']:
        """ Returns the 3x3 rotation matrix R and translation vector t that describe the rigid transform from this frame
        to the given frame.

        The returned R and t describe the rigid transform of a point from this frame ('a') to the given frame ('b')
        by pb = Rpa + t, where pa is the point's coordinate in this frame and pb is its coordinate in the given frame.

        :param frame: The destination frame of the rigid transform.
        :return: Tuple containing the rigid transform's rotation matrix and translation vector.
        """
        return Frame._rigidTransform(self, frame)

    def rigidTransformFrom(self, frame: 'Frame') -> Tuple['Mat3', 'Vec3']:
        """ Returns the 3x3 rotation matrix R and translation vector t that describe the rigid transform to this frame
        from the given frame.

        The returned R and t describe the rigid transform of a point to this frame ('a') from the given frame ('b')
        by pa = Rpb + t, where pa is the point's coordinate in this frame and pb is its coordinate in the given frame.

        :param frame: The origin frame of the rigid transform.
        :return: Tuple containing the rigid transform's rotation matrix and translation vector.
        """
        return Frame._rigidTransform(frame, self)

    def fromWorldToSpherical(self, point: 'Vec3') -> Tuple[_fnp, _fnp, _fnp]:
        """Returns the spherical polar coordinate of a point in this frame, given its 3D coordinate in the world frame.

        :param point: cartesian point in world frame.
        :return: radius, azimuth, polar tuple of spherical polar coordinate in this frame.
        """
        point = self.fromWorld(point)
        return Frame.toSpherical(point)

    def fromSphericalToWorld(self, radius: _fnp, azimuth: _fnp, polar: _fnp) -> 'Vec3':
        """Returns the world cartesian coordinate of a point given its spherical coordinate in this frame.

        :param radius: spherical radius of point.
        :param azimuth: spherical azimuth angle (in radians) of point, measured from x-axis around z-axis.
        :param polar: spherical polar angle (in radians) of point, measured from z-axis.
        :return: cartesian coordinate with respect to world frame.
        """
        return self.toWorld(Frame.fromSpherical(radius, azimuth, polar))

    def translated(self, vector: 'Vec3') -> 'Frame':
        """ Returns this frame linearly translated by the input vector.

        :param vector: the vector by which to shift the frame's origin
        :return: The shifted frame
        """
        return Frame(self._u, self._v, self._w, self._origin + vector)

    def rotatedInPlace(self, around: 'Vec3', by: _fnp) -> 'Frame':
        """ Returns this frame rotated about its origin, anticlockwise around given vector by given amount (in radians).

        :param around: vector around which frame is rotated
        :param by: angle [radians] through which frame is rotated
        :return: the rotated frame
        """
        u = self._u.rotated(around, by)
        v = self._v.rotated(around, by)
        w = self._w.rotated(around, by)
        return Frame(u, v, w, self._origin)

    def rotatedInPlaceByMatrix(self, m: 'Mat3') -> 'Frame':
        """Returns this frame rotated about its origin according to the given rotation matrix

        :param m: the rotation matrix
        :return: the rotated frame
        """
        return Frame(m * self._u, m * self._v, m * self._w, self._origin)

    def rotatedInPlaceByQuaternion(self, q: 'Quaternion') -> 'Frame':
        """Returns this frame rotated about its origin according to the given rotation quaternion

        :param q: the rotation quaternion
        :return: the rotated frame
        """
        m = Mat3.fromQuaternion(q)
        return self.rotatedInPlaceByMatrix(m)

    def rotated(self, point: 'Vec3', axis: 'Vec3', by: _fnp) -> 'Frame':
        """Returns this frame rotated anticlockwise about a given point and rotation axis by a given amount (in radians)

        :param point: the centre of rotation
        :param axis: the axis of rotation
        :param by: the angle through which the frame is rotated anticlockwise [radians]
        :return: the rotated frame
        """
        u = self._u.rotated(axis, by)
        v = self._v.rotated(axis, by)
        w = self._w.rotated(axis, by)
        r = (self.origin - point).rotated(axis, by)
        origin = point + r
        return Frame(u, v, w, origin)

    def quaternionTo(self, other: 'Frame') -> 'Quaternion':
        """Returns the quaternion required to rotate this coordinate frame to the same orientation as another

        :param other: the other frame onto which this frame would be rotated
        :return: the rotation quaternion
        """
        return Frame._frameToFrameQuaternion(self, other)

    def __str__(self):
        return 'u={0}, v={1}, w={2}, origin={3}'.format(self._u, self._v, self._w, self._origin)

    def __repr__(self):
        return 'Frame: u={0}, v={1}, w={2}, origin={3}.'.format(self._u, self._v, self._w, self._origin)

    @property
    def u(self) -> 'Vec3':
        """The frame's u axis"""
        return self._u

    @property
    def v(self) -> 'Vec3':
        """The frame's v axis"""
        return self._v

    @property
    def w(self) -> 'Vec3':
        """The frame's w axis"""
        return self._w

    @property
    def origin(self) -> 'Vec3':
        """The frame's origin"""
        return self._origin

    @origin.setter
    def origin(self, new: 'Vec3'):
        """Updates this frame's origin to the given vector.
        """
        self._origin = new

    @property
    def axes(self) -> Tuple['Vec3', 'Vec3', 'Vec3']:
        """The u, v and w axes of this frame in a tuple"""
        return self._u, self._v, self._w

    @axes.setter
    def axes(self, new: Tuple['Vec3', 'Vec3', 'Vec3']):
        """ Updates this frame's axes to the three given vectors. The given vectors must be normalised and mutually
        perpendicular.

        :param new: tuple containing three normalised, mutually-perpendicular vectors describing this frame's new
            u, v and w axes respectively.
        """
        self._u, self._v, self._w = new
        top = (self._u[0], self._v[0], self._w[0])
        middle = (self._u[1], self._v[1], self._w[1])
        bottom = (self._u[2], self._v[2], self._w[2])
        self._matrix = Mat3(top, middle, bottom)
        self._matrixInverse = self.matrix.transpose

    @property
    def matrix(self) -> 'Mat3':
        """The frame's matrix.
        The matrix's left, middle and right columns are the vectors of the u, v and w axes respectively."""
        return self._matrix

    @property
    def matrixInverse(self) -> 'Mat3':
        """The inverse of this frame's matrix"""
        return self._matrixInverse
