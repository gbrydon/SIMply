# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module containing classes for describing vectors, matrices and quaternions."""

import math
import numpy as np
from typing import Optional, Tuple, Union, List

FloatOrNP = Union[float, np.ndarray]
BoolOrNP = Union[bool, np.ndarray]
Tuple3 = Tuple[FloatOrNP, FloatOrNP, FloatOrNP]


class Vec2:
    """A 2d vector.

    The components (x and y) of this vector can be either floats or numpy arrays (for representing multiple vectors in
    parallel using a single Vec2 object).
    If more than one component is a numpy array, they must have the same shape.
    """
    __array_priority__ = 1000

    def __init__(self, xy: Tuple[FloatOrNP, FloatOrNP]):
        """ Initialises a 2D vector from the given tuple containing x and y components of the vector.

        :param xy: The x and y components of the vector as a 2-element tuple. Each component can be either a float
        or numpy array (if both components are numpy arrays, they must have the same shape).
        """
        self._x = xy[0]
        self._y = xy[1]

    @classmethod
    def zero(cls) -> 'Vec2':
        """ A 2D vector with both components equal to zero"""
        return cls((0, 0))

    @classmethod
    def i(cls) -> 'Vec2':
        """ A unit vector in the x direction"""
        return cls((1, 0))

    @classmethod
    def j(cls) -> 'Vec2':
        """ A unit vector in the y direction"""
        return cls((0, 1))

    @classmethod
    def fromNumpyArray(cls, array: np.ndarray) -> 'Vec2':
        """ Takes a numpy array representation of 2D vectors and returns a Vec2 object. (this is the reverse of the
        Vec2.asNumpyArray property).

        The numpy array must have n dimensions where n >=2 and the innermost axis must be of length 2, e.g.
        [[x1, y1], [x2, y2]...]. In other words, the given array must have a shape of s + (2,) where s is a tuple
        of length >=1. The resulting Vec2 will have a numpy shape (see numpyShape property) of s.

        :param array: the numpy array
        :return: the Vec2 object
        """
        n = array.ndim - 1
        baseSlice = (slice(None),) * n
        x = array[baseSlice + (0,)]
        y = array[baseSlice + (1,)]
        return cls((x, y))

    @property
    def x(self) -> FloatOrNP:
        """The x component of the vector"""
        return self._x

    @x.setter
    def x(self, new: FloatOrNP):
        """Updates the x component of the vector"""
        self._x = new

    @property
    def y(self) -> FloatOrNP:
        """The y component of the vector"""
        return self._y

    @y.setter
    def y(self, new: FloatOrNP):
        """Updates the y component of the vector"""
        self._y = new

    @property
    def length(self) -> FloatOrNP:
        """ The length of the vector"""
        return self.dot(self) ** 0.5

    @property
    def norm(self) -> 'Vec2':
        """ This vector scaled to have a length 1"""
        try:
            return self / self.length
        except ZeroDivisionError:
            return self

    @property
    def tuple(self) -> Tuple[FloatOrNP, FloatOrNP]:
        """ A tuple containing the components of this vector"""
        return self._x, self._y

    @property
    def asNumpyArray(self) -> np.ndarray:
        """Returns this vector as a numpy array (this is the reverse of the Vec2.fromNumpyArray property).

        If this vector is not of numpy type (i.e. none of its components are numpy arrays), this property returns a
        numpy array of [x, y].

        If this vector is of numpy type with numpy shape s, this property returns a numpy array of shape s + (2,) where
        the innermost axis stores the x and y values (e.g. for a vector with numpy shape N, the array takes the form
        [[x1, y1], [x2, y2]...[xN, yN]] and has shape Nx2).

        For example, for a vector of numpy shape (100,), this property
        would return a numpy array of shape (100, 2). For a vector of numpy shape (100, 100) this property would
        return a numpy array of shape (100, 100, 2).
        """
        if self.isNumpyType:
            shape = self.numpyShape
            baseSlice = (slice(None),) * len(shape)
            arr = np.empty(shape + (2,), dtype=float)
            arr[baseSlice + (0,)] = self.x
            arr[baseSlice + (1,)] = self.y
            return arr
        else:
            return np.array(self.tuple)

    @property
    def isNumpyType(self) -> bool:
        """ Whether any of this vector's components is a numpy array. A numpy-type Vec2 can be used to represent
        multiple vectors in parallel using a single Vec2 object."""
        return type(self._x) is np.ndarray or type(self._y) is np.ndarray

    @property
    def numpyShape(self):
        """ If this vector is numpy type (i.e. any of its components is a numpy array), this function returns the shape
        of the numpy array (if a vector is numpy type, all components that are numpy arrays must have the same shape).

        :return: The numpy array shape.
        """
        if type(self._x) is np.ndarray:
            return self._x.shape
        elif type(self._y) is np.ndarray:
            return self._y.shape
        raise TypeError('Tried to access numpy shape of a vector which has no numpy components')

    def dot(self, other: 'Vec2') -> FloatOrNP:
        """ Returns the dot product of this vector with another vector

        :param other: the other vector
        :return: the dot product
        """
        return self._x * other.x + self._y * other.y

    def hadmul(self, other: 'Vec2') -> 'Vec2':
        """ Returns the elementwise Hadamard multiplication of this vector with another vector

        :param other: vector to elementwise multiply with this vector
        :return: the Hadamard product
        """
        return Vec2((self.x * other.x, self.y * other.y))

    def npMask(self, mask: np.ndarray) -> 'Vec2':
        """ Returns the numpy-masked version of this vector.
        The x and y components are each masked and a new vector is formed from the results.
        Should only be used if the vector is numpy-type.

        :param mask: the mask
        :return: the masked vector
        """
        x = self._x
        if type(self._x) is np.ndarray:
            x = self._x[mask]
        y = self._y
        if type(self._y) is np.ndarray:
            y = self._y[mask]
        return Vec2((x, y))

    def __eq__(self, other: 'Vec2'):
        if self.isNumpyType:
            return np.all(np.equal(self._x, other.x) * np.equal(self._y, other.y))
        return self._x == other.x and self._y == other.y

    def __ne__(self, other: 'Vec2'):
        return self._x != other.x or self._y != other.y

    def __abs__(self):
        return self.length

    def __neg__(self):
        return Vec2((-self._x, -self._y))

    def __add__(self, other: 'Vec2'):
        return Vec2((self._x + other.x, self._y + other.y))

    def __radd__(self, other: 'Vec2'):
        return self + other

    def __sub__(self, other: 'Vec2'):
        return self + -other

    def __rsub__(self, other: 'Vec2'):
        return -self + other

    def __mul__(self, value: FloatOrNP):
        return Vec2((self._x * value, self._y * value))

    def __rmul__(self, value: FloatOrNP):
        return self * value

    def __truediv__(self, value: FloatOrNP):
        return Vec2((self._x / value, self._y / value))

    def __setitem__(self, index: int, value: FloatOrNP):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError('Vec2 index must be 0 or 1, not {0}'.format(index))

    def __getitem__(self, index: int) -> FloatOrNP:
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        else:
            raise IndexError('Vec2 index must be 0 or 1, not {0}'.format(index))

    def __str__(self):
        return str(self.tuple)

    def __repr__(self):
        return 'Vec2{0}'.format(self)


class Vec3:
    """A 3d vector.

    The components (x, y and z) of this vector can be either floats or numpy arrays (for representing multiple vectors
    in parallel using a single Vec3 object).
    If more than one component is a numpy array, they must have the same shape.
    """
    __array_priority__ = 1000

    def __init__(self, xyz: Tuple[FloatOrNP, FloatOrNP, FloatOrNP]):
        """ Initialises a 3D vector from a given 1x3 tuple containing the x, y and z components of the vector.

        :param xyz: The x, y and z components of the vector as a 3-element tuple. Each component can be either a float
                    or numpy array (if multiple components are numpy arrays, they must all have the same shape).
        """
        self._x: FloatOrNP = xyz[0]
        self._y: FloatOrNP = xyz[1]
        self._z: FloatOrNP = xyz[2]

    @classmethod
    def zero(cls) -> 'Vec3':
        """ A vector with all components equal to zero"""
        return cls((0, 0, 0))

    @classmethod
    def i(cls) -> 'Vec3':
        """ A unit vector in the x direction"""
        return cls((1, 0, 0))

    @classmethod
    def j(cls) -> 'Vec3':
        """ A unit vector in the y direction"""
        return cls((0, 1, 0))

    @classmethod
    def k(cls) -> 'Vec3':
        """ A unit vector in the z direction"""
        return cls((0, 0, 1))

    @classmethod
    def vectorPerpendicularTo(cls, reference: 'Vec3') -> Optional['Vec3']:
        """ Returns an arbitrary, normalised vector that is perpendicular to the given reference vector

        :param reference: the reference vector
        :return: the arbitrary vector that is perpendicular to the given vector
        """
        if reference.isNumpyType:
            x = (-reference.y - reference.z) / reference.x
            result = Vec3((x, 1, 1)).norm
            inXPlane = reference.x == 0
            result.x[inXPlane] = 1
            result.y[inXPlane] = 0
            result.z[inXPlane] = 0
            return result
        else:
            try:
                x = (-reference.y - reference.z) / reference.x
                return cls((x, 1, 1)).norm
            except ZeroDivisionError:
                try:
                    y = (-reference.x - reference.z) / reference.y
                    return cls((1, y, 1)).norm
                except ZeroDivisionError:
                    try:
                        z = (-reference.x - reference.y) / reference.z
                        return cls((1, 1, z)).norm
                    except ZeroDivisionError:
                        return None

    @classmethod
    def fromNumpyArray(cls, array: np.ndarray) -> 'Vec3':
        """ Takes a numpy array representation of 3D vectors and returns a Vec3 object. (this is the reverse of the
        Vec3.asNumpyArray property).

        The numpy array must have n dimensions where n >=2 and the innermost axis must be of length 3, e.g.
        [[x1, y1, z1], [x2, y2, z2]...]. In other words, the given array must have a shape of s + (3,) where s is a
        tuple of length >=1. The resulting Vec3 will have a numpy shape (see numpyShape property) of s.

        :param array: the numpy array
        :return: the Vec3 object
        """
        n = array.ndim - 1
        baseSlice = (slice(None),) * n
        x = array[baseSlice + (0,)]
        y = array[baseSlice + (1,)]
        z = array[baseSlice + (2,)]
        return cls((x, y, z))

    @property
    def x(self) -> FloatOrNP:
        """The x component of the vector"""
        return self._x

    @x.setter
    def x(self, new: FloatOrNP):
        """Sets the vector's x component to the given value"""
        self._x = new

    @property
    def y(self) -> FloatOrNP:
        """The y component of the vector"""
        return self._y

    @y.setter
    def y(self, new: FloatOrNP):
        """Sets the vector's y component to the given value"""
        self._y = new

    @property
    def z(self) -> FloatOrNP:
        """The z component of the vector"""
        return self._z

    @z.setter
    def z(self, new: FloatOrNP):
        """Sets the vector's z component to the given value"""
        self._z = new

    @property
    def length(self) -> FloatOrNP:
        """ The length of the vector"""
        return self.dot(self) ** 0.5

    @property
    def norm(self) -> 'Vec3':
        """ This vector scaled to have a length 1.

        Where this vector has length zero the same vector will be returned"""
        if self.isNumpyType:
            length = self.length
            mask = length == 0
            normed = self / length
            normed.x[mask] = 0
            normed.y[mask] = 0
            normed.z[mask] = 0
            return normed
        else:
            if self.x == 0 and self.y == 0 and self.z == 0:
                return self
            return self / self.length

    @property
    def tuple(self) -> Tuple[FloatOrNP, FloatOrNP, FloatOrNP]:
        """ A tuple containing the components of this vector"""
        return self._x, self._y, self._z

    @property
    def asNumpyArray(self) -> np.ndarray:
        """Returns this vector as a numpy array (this is the reverse of the Vec3.fromNumpyArray property).

        If this vector is not of numpy type (i.e. none of its components are numpy arrays), this property returns a
        numpy array of [x, y, z].

        If this vector is of numpy type with numpy shape s, this property returns a numpy array of shape s + (3,) where
        the innermost axis stores the x, y and z values (e.g. for a vector with numpy shape N, the array takes the form
        [[x1, y1, z1], [x2, y2, z2]...[xN, yN, zN]] and has shape Nx3).

        For example, for a vector of numpy shape (100,), this property
        would return a numpy array of shape (100, 3). For a vector of numpy shape (100, 100) this property would
        return a numpy array of shape (100, 100, 3).
        """
        if self.isNumpyType:
            shape = self.numpyShape
            baseSlice = (slice(None),) * len(shape)
            arr = np.empty(shape + (3,), dtype=float)
            arr[baseSlice + (0,)] = self.x
            arr[baseSlice + (1,)] = self.y
            arr[baseSlice + (2,)] = self.z
            return arr
        else:
            return np.array(self.tuple)

    @property
    def ontoXY(self) -> 'Vec3':
        """ This vector projected onto the x-y plane"""
        return Vec3((self._x, self._y, 0))

    @property
    def ontoXZ(self) -> 'Vec3':
        """ This vector projected onto the x-z plane"""
        return Vec3((self._x, 0, self._z))

    @property
    def ontoYZ(self) -> 'Vec3':
        """ This vector projected onto the y-z plane"""
        return Vec3((0, self._y, self._z))

    @property
    def isNumpyType(self) -> bool:
        """ Whether any of this vector's components is a numpy array. A numpy-type Vec3 can be used to represent
        multiple vectors in parallel using a single Vec3 object."""
        return type(self._x) is np.ndarray or type(self._y) is np.ndarray or type(self._z) is np.ndarray

    @property
    def numpyShape(self):
        """ If this vector is numpy type (i.e. any of its components is a numpy array), this function returns the shape
        of the numpy array (if a vector is numpy type, all components that are numpy arrays must have the same shape).

        :return: The numpy array shape.
        """
        if type(self._x) is np.ndarray:
            return self._x.shape
        elif type(self._y) is np.ndarray:
            return self._y.shape
        elif type(self._z) is np.ndarray:
            return self._z.shape
        raise TypeError('Tried to access numpy shape of a vector which has no numpy components')

    @property
    def isNan(self):
        """To be used on a numpy-type Vec3: returns boolean array of which vectors are nan (i.e. have component(s) that
        are nan"""
        return np.isnan(self._x) + np.isnan(self._y) + np.isnan(self._z)

    def dot(self, other: 'Vec3') -> FloatOrNP:
        """ Returns the dot product of this vector with another vector

        :param other: the other vector
        :return: the dot product
        """
        return self._x * other.x + self._y * other.y + self._z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        """ Returns the cross product of this vector with another vector

        :param other: the other vector
        :return: the cross product
        """
        a = self._y * other.z - self._z * other.y
        b = self._z * other.x - self._x * other.z
        c = self._x * other.y - self._y * other.x
        return Vec3((a, b, c))

    def hadmul(self, other: 'Vec3') -> 'Vec3':
        """ Returns the element-wise Hadamard multiplication of this vector with another vector

        :param other: vector to element-wise multiply with this vector
        :return: the Hadamard product
        """
        return Vec3((self.x * other.x, self.y * other.y, self.z * other.z))

    def rotated(self, around: 'Vec3', by: FloatOrNP) -> 'Vec3':
        """ Returns this vector rotated anticlockwise around the input vector by given amount (in radians).

        :param around: axis of anticlockwise rotation
        :param by: angle of rotation (radians)
        :return: the rotated vector
        """
        axis = around.norm
        angle = by
        W = Mat3((0, -1 * axis.z, axis.y), (axis.z, 0, -1 * axis.x), (-1 * axis.y, axis.x, 0))
        T1 = np.sin(angle) * W
        T2 = 2 * (np.sin(angle * 0.5) ** 2) * W * W
        M = Mat3.identity() + T1 + T2
        return M * self

    def rotatedByQuaternion(self, q: 'Quaternion') -> 'Vec3':
        """Rotates this vector according to the given rotation quaternion

        :param q: The rotation quaternion
        :return: The rotated vector
        """
        return Vec3((q * Quaternion(0, self.x, self.y, self.z) * q.inverse).tuple[1:])

    def angleWith(self, other: 'Vec3') -> FloatOrNP:
        """ Returns inner angle of this vector with another.

        :param other: the vector forming an angle with this vector
        :return: the inner angle between the two vectors [radians]
        """
        if self.isNumpyType or other.isNumpyType:
            dotResult = self.norm.dot(other.norm)
            dotResult[dotResult < -1] = -1
            dotResult[dotResult > 1] = 1
            return np.arccos(dotResult)
        else:
            try:
                return math.acos(self.norm.dot(other.norm))
            except ValueError:
                return 0

    def signedAngleWith(self, other: 'Vec3', axis: 'Vec3') -> FloatOrNP:
        """ Returns the angle, from -pi to pi, between this vector and another, measured around a mutually
        perpendicular rotation axis.

        :param other: the vector forming an angle with this vector
        :param axis: the axis around which the angle is measured (must be perpendicular to both vectors)
        :return: the signed angle [-pi to pi] between the two vectors [radians]
        """
        b = other
        n = axis.norm
        dot = self.dot(other)
        det = self._x * b.y * n.z + self._z * b.x * n.y + self._y * b.z * n.x - self._z * b.y * n.x - self._x * b.z * n.y - self._y * b.x * n.z
        try:
            return math.atan2(det, dot)
        except TypeError:
            return np.arctan2(det, dot)

    def anticlockAngleWith(self, other: 'Vec3', axis: 'Vec3' = None) -> Tuple[FloatOrNP, 'Vec3']:
        """ Returns the anticlockwise angle from this vector to another, and the mutually perpendicular unit vector
        around which the angle is measured.

        :param other: the vector forming an angle with this vector
        :param axis: the axis around which the angle is measured (must be normalised and perpendicular to both vectors).
                    If None is provided, the normalised cross product of this vector with the other is used.
        :return: the anticlockwise angle from this vector to the other [radians], and the vector around which this angle
                is measured (provided as an [angle, vector tuple])
        """
        if axis is None:
            axis = self.cross(other).norm
        if self.isNumpyType or other.isNumpyType or axis.isNumpyType:
            return np.arctan2(self.cross(other).dot(axis), self.dot(other)), axis
        else:
            return math.atan2(self.cross(other).dot(axis), self.dot(other)), axis

    def quaternionTo(self, other: 'Vec3') -> 'Quaternion':
        """ Returns the rotation quaternion needed to rotate this vector onto another

        :param other: the other vector
        :return: the rotation quaternion
        """
        ang, ax = self.anticlockAngleWith(other)
        return Quaternion.fromRotationParams(ax, ang)

    def isParallelWith(self, other: 'Vec3', epsilon=1e-7) -> BoolOrNP:
        """ Whether this vector is parallel with, and points the same direction as, another vector

        :param other: the other vector
        :param epsilon: If the angle between the two vectors is below this threshold, this function returns true
        :return: whether the vectors are parallel
        """
        return self.angleWith(other) < epsilon

    def isAntiparallelWith(self, other: 'Vec3', epsilon=1e-7) -> BoolOrNP:
        """ Whether this vector is antiparallel (i.e. they are parallel but point opposite directions) with
        another vector

        :param other: the other vector
        :param epsilon: If the angle between the two vectors, when shifted by pi, is below this threshold, this function
                        returns true
        :return: whether the vectors are antiparallel
        """
        return self.angleWith(other) > math.pi - epsilon

    def projectedOnto(self, other: 'Vec3') -> 'Vec3':
        """ Returns the result of projecting this vector onto another vector
        (i.e. it returns the component of this vector that is parallel to the other vector)

        :param other: the other vector
        :return: the parallel projection
        """
        return other * self.dot(other) / (other.length ** 2)

    def projectedPerpTo(self, other: 'Vec3') -> 'Vec3':
        """ Returns the component of this vector that is perpendicular to another vector

        :param other: the other vector
        :return: the perpendicular projection
        """
        return self - self.projectedOnto(other)

    def npMask(self, mask: np.ndarray) -> 'Vec3':
        """ Returns the numpy-masked version of this vector.
        The x, y and z components are each masked and a new vector is formed from the results.

        :param mask: the mask
        :return: the masked vector
        """
        x = self._x
        if type(self._x) is np.ndarray:
            x = self._x[mask]
        y = self._y
        if type(self._y) is np.ndarray:
            y = self._y[mask]
        z = self._z
        if type(self._z) is np.ndarray:
            z = self._z[mask]
        return Vec3((x, y, z))

    def isClose(self, other: 'Vec3', rtol: float = 1e-5, atol: float = 1e-8) -> BoolOrNP:
        """ Returns whether the element-wise comparison of this vector with another is equal within some tolerance (i.e.
        the x, y and z components of this vector are sufficiently close to the x, y and z components respectively of
        the other vector).

        If either of the vectors contain components represented by numpy arrays, these components are compared
        element-wise and this function returns a numpy array of matching shape.

        Uses numpy's isclose function - see associated documentation.

        :param other: the other vector
        :param rtol: the allowable relative difference
        :param atol: the allowable absolute difference
        :return: whether the two vectors' components are sufficiently close.
        """
        x = np.isclose(self.x, other.x, rtol, atol)
        y = np.isclose(self.y, other.y, rtol, atol)
        z = np.isclose(self.z, other.z, rtol, atol)
        return x * y * z

    def allClose(self, other: 'Vec3', rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """ For if this vector or the given vector conform to .isNumpyType == True, this function returns whether all
        elements of each component of this vector compare element-wise to all corresponding elements of the other vector
        within some tolerance.

        Uses numpy's allclose function - see associated documentation

        :param other: the other vector
        :param rtol: the allowable relative difference
        :param atol: the allowable absolute difference
        :return: whether the two vectors' corresponding elements all match within the given tolerance
        """
        x = np.allclose(self.x, other.x, rtol, atol)
        y = np.allclose(self.y, other.y, rtol, atol)
        z = np.allclose(self.z, other.z, rtol, atol)
        return x * y * z

    def __eq__(self, other: 'Vec3'):
        if self.isNumpyType:
            return np.equal(self._x, other.x) * np.equal(self._y, other.y) * np.equal(self._z, other.z)
        return self._x == other.x and self._y == other.y and self._z == other.z

    def __ne__(self, other: 'Vec3'):
        return self._x != other.x or self._y != other.y or self._z != other.z

    def __abs__(self):
        return self.length

    def __neg__(self):
        return Vec3((-self._x, -self._y, -self._z))

    def __add__(self, other: 'Vec3'):
        return Vec3((self._x + other.x, self._y + other.y, self._z + other.z))

    def __radd__(self, other: 'Vec3'):
        return self + other

    def __sub__(self, other: 'Vec3'):
        return self + -other

    def __rsub__(self, other: 'Vec3'):
        return -self + other

    def __mul__(self, value: FloatOrNP):
        return Vec3((self._x * value, self._y * value, self._z * value))

    def __rmul__(self, value: FloatOrNP):
        return self * value

    def __truediv__(self, value: FloatOrNP):
        return Vec3((self._x / value, self._y / value, self._z / value))

    def __setitem__(self, index: int, value: FloatOrNP):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError('Vec3 index must be 0, 1 or 2, not {0}'.format(index))

    def __getitem__(self, index: int) -> FloatOrNP:
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        elif index == 2:
            return self._z
        else:
            raise IndexError('Vec3 index must be 0, 1 or 2, not {0}'.format(index))

    def __str__(self):
        return str(self.tuple)

    def __repr__(self):
        return 'Vec3{0}'.format(self)


class Mat3:
    """A 3x3 matrix.
    The elements of this matrix can be floats or numpy arrays (for representing multiple matrices in parallel using a
    single Mat3 object).
    If more than one element is a numpy array, they must have the same shape.
    """

    __array_priority__ = 1000

    def __init__(self, row1: Tuple3, row2: Tuple3, row3: Tuple3):
        """ Initialises a new 3x3 matrix with the given rows of elements.

        :param row1: The top row of the matrix.
        :param row2: The middle row of the matrix.
        :param row3: The bottom row of the matrix.
        """
        if row1 is None:
            row1 = (0, 0, 0)
        if row2 is None:
            row2 = (0, 0, 0)
        if row3 is None:
            row3 = (0, 0, 0)
        self.x00: FloatOrNP = row1[0]
        self.x01: FloatOrNP = row1[1]
        self.x02: FloatOrNP = row1[2]
        self.x10: FloatOrNP = row2[0]
        self.x11: FloatOrNP = row2[1]
        self.x12: FloatOrNP = row2[2]
        self.x20: FloatOrNP = row3[0]
        self.x21: FloatOrNP = row3[1]
        self.x22: FloatOrNP = row3[2]

    @classmethod
    def identity(cls) -> 'Mat3':
        """The 3x3 identity matrix"""
        return cls((1, 0, 0), (0, 1, 0), (0, 0, 1))

    @classmethod
    def fromColumns(cls, col1: Union[Tuple3, Vec3], col2: Union[Tuple3, Vec3], col3: Union[Tuple3, Vec3]) -> 'Mat3':
        """ Returns a new 3x3 matrix from the given columns of elements.

        :param col1: The left column
        :param col2: The centre column
        :param col3: The right column.
        :return: The 3x3 matrix.
        """
        row1 = (col1[0], col2[0], col3[0])
        row2 = (col1[1], col2[1], col3[1])
        row3 = (col1[2], col2[2], col3[2])
        return cls(row1, row2, row3)

    @classmethod
    def fromRotationParams(cls, axis: 'Vec3', angle: FloatOrNP) -> 'Mat3':
        """ Returns a rotation matrix which performs an anticlockwise rotation around the given axis by the given angle.

        :param axis: a unit vector defining the rotation axis
        :param angle: the rotation angle [radians]
        :return: the rotation matrix
        """
        cos = np.cos(angle)
        sin = np.sin(angle)
        x00 = cos + axis.x * axis.x * (1 - cos)
        x01 = axis.x * axis.y * (1 - cos) - axis.z * sin
        x02 = axis.x * axis.z * (1 - cos) + axis.y * sin
        x10 = axis.x * axis.y * (1 - cos) + axis.z * sin
        x11 = cos + axis.y * axis.y * (1 - cos)
        x12 = axis.y * axis.z * (1 - cos) - axis.x * sin
        x20 = axis.z * axis.x * (1 - cos) - axis.y * sin
        x21 = axis.z * axis.y * (1 - cos) + axis.x * sin
        x22 = cos + axis.z * axis.z * (1 - cos)
        return Mat3((x00, x01, x02), (x10, x11, x12), (x20, x21, x22))

    @classmethod
    def triadRotation(cls, v1a: Vec3, v2a: Vec3, v1b: Vec3, v2b: Vec3) -> 'Mat3':
        """ Given two direction vectors represented in two different coordinate frames, this function returns a matrix
        that rotates any direction vector in frame A to its representation in frame B. This function uses the Triad
        method https://en.wikipedia.org/wiki/Triad_method.

        :param v1a: direction vector 1 in frame A
        :param v2a: direction vector 2 in frame A
        :param v1b: direction vector 1 in frame B
        :param v2b: direction vector 2 in frame B
        :return: matrix that transforms direction vectors from frame A to frame B
        """
        sa = v1a.norm
        sb = v1b.norm

        ma = v1a.cross(v2a).norm
        mb = v1b.cross(v2b).norm

        mat1 = Mat3.fromColumns(sb, mb, sb.cross(mb))
        mat2 = Mat3.fromColumns(sa, ma, sa.cross(ma)).transpose
        return mat1 * mat2

    @classmethod
    def fromQuaternion(cls, q: 'Quaternion'):
        """ Creates a rotation matrix from the given rotation quaternion.

        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        http://www.songho.ca/opengl/gl_quaternion.html

        :param q: the rotation quaternion
        """
        qs, q1, q2, q3 = q.tuple
        x00 = 2 * (qs * qs + q1 * q1) - 1
        x01 = 2 * (q1 * q2 - qs * q3)
        x02 = 2 * (q1 * q3 + qs * q2)
        x10 = 2 * (q1 * q2 + qs * q3)
        x11 = 2 * (qs * qs + q2 * q2) - 1
        x12 = 2 * (q2 * q3 - qs * q1)
        x20 = 2 * (q1 * q3 - qs * q2)
        x21 = 2 * (q2 * q3 + qs * q1)
        x22 = 2 * (qs * qs + q3 * q3) - 1

        # qs, q1, q2, q3 = q.tuple
        # x00 = 1 - 2 * (q2 * q2 + q3 * q3)
        # x01 = 2 * (q1 * q2 - q3 * qs)
        # x02 = 2 * (q1 * q3 + q2 * qs)
        # x10 = 2 * (q1 * q2 + q3 * qs)
        # x11 = 1 - 2 * (q1 * q1 + q3 * q3)
        # x12 = 2 * (q2 * q3 - q1 * qs)
        # x20 = 2 * (q1 * q3 - q2 * qs)
        # x21 = 2 * (q2 * q3 + q1 * qs)
        # x22 = 1 - 2 * (q1 * q1 + q2 * q2)

        return Mat3((x00, x01, x02), (x10, x11, x12), (x20, x21, x22))

    @property
    def determ(self) -> FloatOrNP:
        """The determinant of this matrix"""
        a = self.x00 * ((self.x11 * self.x22) - (self.x12 * self.x21))
        b = self.x01 * ((self.x10 * self.x22) - (self.x12 * self.x20))
        c = self.x02 * ((self.x10 * self.x21) - (self.x11 * self.x20))
        return a - b + c

    @property
    def transpose(self) -> 'Mat3':
        """ Returns the transpose of this matrix.
        The transpose swaps the row and column indices of each element of the matrix.

        :return: The transposed matrix
        """
        top = (self.x00, self.x10, self.x20)
        middle = (self.x01, self.x11, self.x21)
        bottom = (self.x02, self.x12, self.x22)
        return Mat3(top, middle, bottom)

    @property
    def inverse(self) -> 'Mat3':
        """ Calculates and returns the inverse of this matrix using numpy.linalg function.

        :return: The inverse matrix
        """
        a = np.array([[self.x00, self.x01, self.x02], [self.x10, self.x11, self.x12], [self.x20, self.x21, self.x22]])
        b = np.linalg.inv(a)
        return Mat3(b[0], b[1], b[2])

    @property
    def row1(self) -> Tuple3:
        """The top row of the matrix"""
        return self.x00, self.x01, self.x02

    @property
    def row2(self) -> Tuple3:
        """The middle row of the matrix"""
        return self.x10, self.x11, self.x12

    @property
    def row3(self) -> Tuple3:
        """The bottom row of the matrix"""
        return self.x20, self.x21, self.x22

    @property
    def column1(self) -> Tuple3:
        """The left column of the matrix"""
        return self.x00, self.x10, self.x20

    @property
    def column2(self) -> Tuple3:
        """The middle column of the matrix"""
        return self.x01, self.x11, self.x21

    @property
    def column3(self) -> Tuple3:
        """The right column of the matrix"""
        return self.x02, self.x12, self.x22

    @property
    def asLists(self) -> List[List[FloatOrNP]]:
        """The matrix as a list of lists."""
        return [list(self.row1), list(self.row2), list(self.row3)]

    @property
    def asNumpy(self) -> np.ndarray:
        """The matrix as a 3x3 numpy array."""
        return np.array([self.row1, self.row2, self.row3])

    @property
    def elements1d(self) -> Tuple[FloatOrNP, ...]:
        """All of this matrix's elements as a 1d tuple, in row-major order"""
        return self.x00, self.x01, self.x02, self.x10, self.x11, self.x12, self.x20, self.x21, self.x22

    @property
    def isNumpyType(self) -> bool:
        """ Whether any of this matrix's elements is a numpy array"""
        for element in self.elements1d:
            if type(element) is np.ndarray:
                return True
        return False

    @property
    def numpyShape(self):
        """ If this matrix is numpy type (i.e. any of its elements is a numpy array), this function returns the shape
        of the numpy array (if a matrix is numpy type, all elements that are numpy arrays must have the same shape).

        :return: The numpy array shape.
        """
        for element in self.elements1d:
            if type(element) is np.ndarray:
                return element.shape
        raise TypeError('Tried to access numpy shape of a matrix which has no numpy components')

    @property
    def singleLineString(self) -> str:
        """String representation of this matrix containing no line breaks (i.e. occupies a single line)."""
        r1 = [self.x00, self.x01, self.x02]
        r2 = [self.x10, self.x11, self.x12]
        r3 = [self.x20, self.x21, self.x22]
        return '[{0}, {1}, {2}]'.format(r1, r2, r3)

    def hadmul(self, other: 'Mat3') -> 'Mat3':
        """ Returns the hadamard product (element-wise multiplication) of this and another matrix.

        :param other: The other 3x3 matrix
        :return: The hadamard product of the two matrices
        """
        row1 = (self.x00 * other.x00, self.x01 * other.x01, self.x02 * other.x02)
        row2 = (self.x10 * other.x10, self.x11 * other.x11, self.x12 * other.x12)
        row3 = (self.x20 * other.x20, self.x21 * other.x21, self.x22 * other.x22)
        return Mat3(row1, row2, row3)

    def isClose(self, other: 'Mat3', rtol: float = 1e-5, atol: float = 1e-8) -> BoolOrNP:
        """ Returns whether the element-wise comparison of this matrix with another is equal within some tolerance (i.e.
        the elements of this matrix are sufficiently close in value to their counterparts in the other matrix).

        If either of the matrices contain components represented by numpy arrays, these components are compared
        element-wise and this function returns a numpy array of matching shape.

        Uses numpy's isclose function - see associated documentation.

        :param other: the other matrix
        :param rtol: the allowable relative difference
        :param atol: the allowable absolute difference
        :return: whether the two matrices' components are sufficiently close.
        """
        x00 = np.isclose(self.x00, other.x00, rtol, atol)
        x01 = np.isclose(self.x01, other.x01, rtol, atol)
        x02 = np.isclose(self.x02, other.x02, rtol, atol)
        x10 = np.isclose(self.x10, other.x10, rtol, atol)
        x11 = np.isclose(self.x11, other.x11, rtol, atol)
        x12 = np.isclose(self.x12, other.x12, rtol, atol)
        x20 = np.isclose(self.x20, other.x20, rtol, atol)
        x21 = np.isclose(self.x21, other.x21, rtol, atol)
        x22 = np.isclose(self.x22, other.x22, rtol, atol)
        return x00 * x01 * x02 * x10 * x11 * x12 * x20 * x21 * x22

    def allClose(self, other: 'Mat3', rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """ For if this matrix or the given matrix conform to .isNumpyType == True, this function returns whether all
        elements of this matrix compare element-wise to all corresponding elements of the other matrix
        within some tolerance.

        Uses numpy's allclose function - see associated documentation

        :param other: the other matrix
        :param rtol: the allowable relative difference
        :param atol: the allowable absolute difference
        :return: whether the two matrices' corresponding elements all match within the given tolerance
        """
        x00 = np.allclose(self.x00, other.x00, rtol, atol)
        x01 = np.allclose(self.x01, other.x01, rtol, atol)
        x02 = np.allclose(self.x02, other.x02, rtol, atol)
        x10 = np.allclose(self.x10, other.x10, rtol, atol)
        x11 = np.allclose(self.x11, other.x11, rtol, atol)
        x12 = np.allclose(self.x12, other.x12, rtol, atol)
        x20 = np.allclose(self.x20, other.x20, rtol, atol)
        x21 = np.allclose(self.x21, other.x21, rtol, atol)
        x22 = np.allclose(self.x22, other.x22, rtol, atol)
        return x00 * x01 * x02 * x10 * x11 * x12 * x20 * x21 * x22

    def __eq__(self, other: 'Mat3'):
        if self.isNumpyType:
            result = np.full(self.numpyShape, 1)
            for elementA, elementB in zip(self.elements1d, other.elements1d):
                result = result * np.equal(elementA, elementB)
            return result
        for elementA, elementB in zip(self.elements1d, other.elements1d):
            if elementA != elementB:
                return False
        return True

    def __ne__(self, other: 'Mat3'):
        if self.isNumpyType:
            result = np.full(self.numpyShape, 1)
            for elementA, elementB in zip(self.elements1d, other.elements1d):
                result = result * ~np.equal(elementA, elementB)
            return result
        for elementA, elementB in zip(self.elements1d, other.elements1d):
            if elementA != elementB:
                return True
        return False

    def __add__(self, other: 'Mat3') -> 'Mat3':
        t = (self.x00 + other.x00, self.x01 + other.x01, self.x02 + other.x02)
        m = (self.x10 + other.x10, self.x11 + other.x11, self.x12 + other.x12)
        b = (self.x20 + other.x20, self.x21 + other.x21, self.x22 + other.x22)
        return Mat3(t, m, b)

    def __sub__(self, other: 'Mat3') -> 'Mat3':
        t = (self.x00 - other.x00, self.x01 - other.x01, self.x02 - other.x02)
        m = (self.x10 - other.x10, self.x11 - other.x11, self.x12 - other.x12)
        b = (self.x20 - other.x20, self.x21 - other.x21, self.x22 - other.x22)
        return Mat3(t, m, b)

    def __mul__(self, other: Union[FloatOrNP, 'Vec3', 'Mat3']):
        if type(other) is Vec3:
            a = (self.x00 * other.x) + (self.x01 * other.y) + (self.x02 * other.z)
            b = (self.x10 * other.x) + (self.x11 * other.y) + (self.x12 * other.z)
            c = (self.x20 * other.x) + (self.x21 * other.y) + (self.x22 * other.z)
            return Vec3((a, b, c))
        if type(other) is Mat3:
            M = other
            top0 = self.x00 * M.x00 + self.x01 * M.x10 + self.x02 * M.x20
            top1 = self.x00 * M.x01 + self.x01 * M.x11 + self.x02 * M.x21
            top2 = self.x00 * M.x02 + self.x01 * M.x12 + self.x02 * M.x22
            middle0 = self.x10 * M.x00 + self.x11 * M.x10 + self.x12 * M.x20
            middle1 = self.x10 * M.x01 + self.x11 * M.x11 + self.x12 * M.x21
            middle2 = self.x10 * M.x02 + self.x11 * M.x12 + self.x12 * M.x22
            bottom0 = self.x20 * M.x00 + self.x21 * M.x10 + self.x22 * M.x20
            bottom1 = self.x20 * M.x01 + self.x21 * M.x11 + self.x22 * M.x21
            bottom2 = self.x20 * M.x02 + self.x21 * M.x12 + self.x22 * M.x22
            top = (top0, top1, top2)
            middle = (middle0, middle1, middle2)
            bottom = (bottom0, bottom1, bottom2)
            return Mat3(top, middle, bottom)
        row1 = other * self.x00, other * self.x01, other * self.x02
        row2 = other * self.x10, other * self.x11, other * self.x12
        row3 = other * self.x20, other * self.x21, other * self.x22
        return Mat3(row1, row2, row3)

    def __rmul__(self, scalar: FloatOrNP) -> 'Mat3':
        """Returns the result of multiplying a scalar by this matrix."""
        row1 = scalar * self.x00, scalar * self.x01, scalar * self.x02
        row2 = scalar * self.x10, scalar * self.x11, scalar * self.x12
        row3 = scalar * self.x20, scalar * self.x21, scalar * self.x22
        return Mat3(row1, row2, row3)

    def __truediv__(self, value: FloatOrNP) -> 'Mat3':
        multiplier = 1 / value
        return self * multiplier

    def __pow__(self, power: int, modulo=None) -> 'Mat3':
        result = self
        for _ in range(power - 1):
            result = result * self
        return result

    def __str__(self):
        r1 = [self.x00, self.x01, self.x02]
        r2 = [self.x10, self.x11, self.x12]
        r3 = [self.x20, self.x21, self.x22]
        return '[{0},\n{1},\n{2}]'.format(r1, r2, r3)

    def __repr__(self):
        return 'Mat3{0}'.format(self)


class Quaternion:
    """A quaternion."""

    def __init__(self, s: FloatOrNP, x: FloatOrNP, y: FloatOrNP, z: FloatOrNP):
        """ Initialises a new quaternion

        :param s: The quaternion's scalar value
        :param x: The quaternion's first complex/vector value
        :param y: The quaternion's second complex/vector value
        :param z: The quaternion's third complex/vector value
        """
        self._s = s  # the scalar value
        self._x = x  # the first complex/vector value
        self._y = y  # the second complex/vector value
        self._z = z  # the third complex/vector value

    @classmethod
    def fromRotationParams(cls, axis: 'Vec3', angle: FloatOrNP) -> 'Quaternion':
        """Returns a rotation quaternion that rotates a vector anticlockwise around the given axis by the given angle
        [radians]

        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html

        :param axis: unit vector defining the rotation axis
        :param angle: the rotation angle [radians]
        :return: the quaternion
        """
        axis = axis.norm
        s = np.cos(0.5 * angle)
        m = np.sin(0.5 * angle)
        x = axis.x * m
        y = axis.y * m
        z = axis.z * m
        return cls(s, x, y, z)

    @property
    def s(self) -> FloatOrNP:
        """The quaternion's scalar value"""
        return self._s

    @s.setter
    def s(self, new: FloatOrNP):
        """Sets the quaternion's scalar value to the given value"""
        self._s = new

    @property
    def x(self) -> FloatOrNP:
        """The quaternion's x value (1st vector value)"""
        return self._x

    @x.setter
    def x(self, new: FloatOrNP):
        """Sets the quaternion's x value (1st vector value) to the given value"""
        self._x = new

    @property
    def y(self) -> FloatOrNP:
        """The quaternion's y value (2nd vector value)"""
        return self._y

    @y.setter
    def y(self, new: FloatOrNP):
        """Sets the quaternion's y value (2nd vector value) to the given value"""
        self._y = new

    @property
    def z(self) -> FloatOrNP:
        """The quaternion's z value (3rd vector value)"""
        return self._z

    @z.setter
    def z(self, new):
        """Sets the quaternion's z value (3rd vector value) to the given value"""
        self._z = new

    @property
    def normValue(self) -> FloatOrNP:
        """The scalar norm (Euclidean norm) of the quaternion"""
        return (self._s ** 2 + self._x ** 2 + self._y ** 2 + self._z ** 2) ** 0.5

    @property
    def unit(self) -> 'Quaternion':
        s = 1 / self.normValue
        return Quaternion(self._s * s, self._x * s, self._y * s, self._z * s)

    @property
    def inverse(self) -> 'Quaternion':
        """The inverse of this quaternion (corresponding to the reverse rotation)"""
        return Quaternion(self._s, -self._x, -self._y, -self._z)

    @property
    def tuple(self) -> Tuple[FloatOrNP, FloatOrNP, FloatOrNP, FloatOrNP]:
        """The quaternion's values as a tuple (with scalar value leading)"""
        return self._s, self._x, self._y, self._z

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        assert(type(other) is Quaternion)
        s = self.s * other.s - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.s * other.x + self.x * other.s + self.y * other.z - self.z * other.y
        y = self.s * other.y - self.x * other.z + self.y * other.s + self.z * other.x
        z = self.s * other.z + self.x * other.y - self.y * other.x + self.z * other.s
        return Quaternion(s, x, y, z)

    def __str__(self):
        return str(self.tuple)

    def __repr__(self):
        return 'Quaternion{0}'.format(self)

