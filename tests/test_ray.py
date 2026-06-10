import numpy as np
from coremaths.vector import Vec3
from coremaths.ray import Ray


def rand(n, s=10):
    return np.random.normal(0, s, n)


def test_pointsWithDistanceFromArbitrary():
    n = 1000
    rayOrigin = Vec3((rand(n), rand(n), rand(n)))
    rayDir = Vec3((rand(n), rand(n), rand(n))).norm
    refPoint = Vec3((rand(n), rand(n), rand(n)))

    ray = Ray(rayOrigin, rayDir)

    refDistance = 400
    points = ray.pointsAtGivenDistanceFromArbitrary(refPoint, refDistance)
    p1, p2 = points

    d1 = (p1 - refPoint).length
    d2 = (p2 - refPoint).length

    d1 = d1[~np.isnan(d1)]
    d2 = d1[~np.isnan(d2)]

    assert np.allclose(d1, refDistance)
    assert np.allclose(d2, refDistance)
