# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module providing functionality for describing rays (straight lines with direction and point of
origin in 3D space).
"""

import numpy as np
from coremaths.vector import Vec3
from coremaths.frame import Frame
from typing import Union

FloatOrNp = Union[float, np.ndarray]


class Ray:
    """A ray defined by a straight line originating from a point in 3D space."""
    def __init__(self, origin: 'Vec3', direction: 'Vec3'):
        """ Initialises a ray defining a line originating at the given origin and travelling in the given direction.

        Both origin and direction vectors can be of numpy-type (see Vec3). If they are both numpy-type, they must
        have the same shape.

        :param origin: the origin of the ray
        :param direction: Normalised vector aligned with the ray's direction.
        """
        self._origin: 'Vec3' = origin
        self._d: 'Vec3' = direction

    @property
    def origin(self):
        """The position of the ray's origin (i.e. the point along the ray with zero depth)"""
        return self._origin

    @origin.setter
    def origin(self, point: 'Vec3'):
        """Sets the ray's origin to the given point"""
        self._origin = point

    @property
    def d(self):
        """A normalised vector describing the direction of the ray"""
        return self._d

    @d.setter
    def d(self, n: 'Vec3'):
        """Sets this ray's direction vector to the given normalised vector"""
        self._d = n

    @property
    def isNumpyType(self):
        """Whether either the origin or direction vector of this ray is numpy-type"""
        return self.origin.isNumpyType or self.d.isNumpyType

    @property
    def numpyShape(self):
        """ If this ray is numpy-type (i.e. either its origin or direction vector is numpy-type)
        this property returns its numpy shape (see Vec3).

        :return: The numpy shape.
        """
        try:
            return self._origin.numpyShape
        except TypeError:
            return self._d.numpyShape

    def point(self, depth: 'FloatOrNp') -> 'Vec3':
        """ Returns the point on this ray with given distance from the ray's origin.

        If this ray and the given depth are both of numpy type, they must have the same shape.

        :param depth: the point's depth (i.e. distance) from the ray's origin
        :return: the point on the ray
        """
        return self.origin + depth * self.d

    def pointWithX(self, x: 'FloatOrNp') -> 'Vec3':
        """ Returns the point on this ray with given x value.

        If this ray and the given x value are both of numpy type, they must have the same shape.

        :param x: the desired x value
        :return: the point on the ray
        """
        depth = (x - self.origin.x) / self.d.x
        return self.point(depth)

    def pointWithY(self, y: 'FloatOrNp') -> 'Vec3':
        """ Returns the point on this ray with given y value.

        If this ray and the given y value are both of numpy type, they must have the same shape.

        :param y: the desired y value
        :return: the point on the ray
        """
        depth = y - self.origin.y / self.d.y
        return self.point(depth)

    def pointWithZ(self, z: 'FloatOrNp') -> 'Vec3':
        """ Returns the point on this ray with given z value.

        If this ray and the given z value are both of numpy type, they must have the same shape.

        :param z: the desired z value
        :return: the point on the ray
        """
        depth = z - self.origin.z / self.d.z
        return self.point(depth)

    def transformed(self, old: 'Frame' = Frame.world(), new: 'Frame' = Frame.world()) -> 'Ray':
        """ Transforms this ray from a given coordinate frame to a new coordinate frame and returns the result.

        :param old: the original 3D cartesian frame from which the ray is being converted
        :param new: the new cartesian frame to which the ray is being converted
        :return: the converted ray, represented in the new frame's coordinates
        """
        p1 = old.toFrame(self.origin, new)
        p2 = old.toFrame(self.point(1), new)
        return Ray(p1, p2 - p1)

    def numpyMasked(self, mask: np.ndarray) -> 'Ray':
        """ Returns the numpy-masked version of this ray (where all numpy components of its vectors are masked by the
        given mask, see documentation for same function in Vec3 class)

        :param mask: the mask
        :return: the masked ray
        """
        return Ray(self.origin.npMask(mask), self.d.npMask(mask))

    def __str__(self):
        return 'origin: {0}, normal: {1}'.format(self.origin, self.d)

    def __repr__(self):
        return 'Ray[{}]'.format(self)
