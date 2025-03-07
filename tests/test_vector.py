# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
import math
import numpy as np
from coremaths.vector import Vec3, Mat3, Quaternion


def rand(n, s=100):
    return np.random.normal(0, s, n)


def test_Vec3Length():
    v = Vec3((-3, 4, 5))
    assert v.length == 50 ** 0.5, "incorrect length property for non-numpy-type Vec3"
    n = 1000
    x = rand(n)
    y = rand(n)
    z = rand(n)
    length = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    v = Vec3((x, y, z))
    assert np.allclose(v.length, length, 1e-8, 0), "incorrect length property for numpy-type Vec3"


def test_Vec3Dot():
    n = 1000
    a = Vec3((rand(n), rand(n), rand(n)))
    b = Vec3((rand(n), rand(n), rand(n)))
    check = a.dot(b)
    truth = a.x * b.x + a.y * b.y + a.z * b.z
    assert np.array_equal(truth, check), 'Vec3 dot product calculated incorrect values'
    assert np.array_equal(check, b.dot(a)), 'Vec3 dot product failed commutativity test (i.e. a.b did not equal b.a)'


def test_Vec3Cross():
    assert Vec3.i().cross(Vec3.j()) == Vec3.k()
    assert Vec3.i().cross(Vec3.k()) == -Vec3.j()
    assert Vec3.j().cross(Vec3.k()) == Vec3.i()
    assert Vec3((4, -9, 13)).cross(Vec3((7, 23, 8))) == -(Vec3((7, 23, 8)).cross(Vec3((4, -9, 13))))
    n = 1000
    v1 = Vec3((rand(n), rand(n), rand(n)))
    v2 = Vec3((rand(n), rand(n), rand(n)))
    check = v1.cross(v2)
    truth = Vec3((v1._y * v2.z - v1._z * v2.y, v1._z * v2.x - v1._x * v2.z, v1._x * v2.y - v1._y * v2.x))
    assert check.allClose(truth, 1e-8, 0)


def test_Vec3InnerAngles():
    v1 = Vec3((4, 7, 2))
    v2 = Vec3((-1, 6, 9))
    check = v1.angleWith(v2)
    truth = math.acos(v1.dot(v2) / (v1.length * v2.length))
    assert truth == check, 'Vec3 inner angle calculated incorrectly for non-numpy Vec3'
    assert Vec3.i().angleWith(Vec3.j()) == math.radians(90)
    assert Vec3.i().angleWith(Vec3.k()) == math.radians(90)
    assert Vec3.k().angleWith(Vec3.j()) == math.radians(90)
    n = 1000
    v1 = Vec3((rand(n), rand(n), rand(n)))
    v2 = Vec3((rand(n), rand(n), rand(n)))
    check = v1.angleWith(v2)
    truth = np.arccos(v1.dot(v2) / (v1.length * v2.length))
    assert np.allclose(check, truth, 1e-8, 0), 'Vec3 inner angle calculated incorrectly for numpy-type Vec3'


def test_Vec3GetPerpendicularVector():
    n = 1000
    v1 = Vec3((rand(n), rand(n), rand(n))).norm
    v2 = Vec3.vectorPerpendicularTo(v1).norm
    assert np.allclose(v1.angleWith(v2), 0.5 * math.pi)
    v1 = Vec3((rand(n), rand(n), rand(n))).norm
    v2 = Vec3((rand(n), rand(n), rand(n))).norm
    v2 = v2.projectedPerpTo(v1)
    assert np.allclose(v1.angleWith(v2), 0.5 * math.pi)


def test_Vec3GetParallelVector():
    n = 1000
    v1 = Vec3((rand(n), rand(n), rand(n))).norm
    v2 = Vec3((rand(n), rand(n), rand(n))).norm
    v2 = v2.projectedOnto(v1)
    assert np.all(v1.isParallelWith(v2) + v1.isAntiparallelWith(v2))


def test_Vec3Rotate():
    r = Vec3.i().rotated(Vec3.k(), 0.5 * math.pi)
    assert (r.allClose(Vec3.j()))
    r = Vec3.j().rotated(Vec3.i(), 0.5 * math.pi)
    assert (r.allClose(Vec3.k()))
    r = Vec3.k().rotated(Vec3.j(), 0.5 * math.pi)
    assert (r.allClose(Vec3.i()))

    def truth(vec: 'Vec3', axis: 'Vec3', angle) -> 'Vec3':
        # Rodrigues' rotation formula (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
        return vec * np.cos(angle) + (axis.cross(vec)) * np.sin(angle) + axis * (axis.dot(vec)) * (1 - np.cos(angle))

    n = 10000

    v = Vec3((rand(n), rand(n), rand(n))).norm
    ax = Vec3((rand(n), rand(n), rand(n))).norm
    ang = np.random.uniform(0, 0.5 * np.pi, n)
    v1 = v.rotated(ax, ang)
    v2 = truth(v, ax, ang)
    assert v1.allClose(v2)

    v1 = Vec3((rand(n), rand(n), rand(n))).norm
    ax = Vec3((rand(n), rand(n), rand(n)))
    ax = ax.vectorPerpendicularTo(v1).norm
    v2 = v1.rotated(ax, math.pi)
    assert v1.allClose(-v2)

    v1 = Vec3((rand(n), rand(n), rand(n))).norm
    ax = Vec3((rand(n), rand(n), rand(n))).norm
    v2 = v1.rotated(ax, 2 * math.pi)
    assert v1.allClose(v2)


def test_Vec3AnticlockwiseAngles():
    n = 10000
    v = Vec3((rand(n), rand(n), rand(n))).norm
    ax = Vec3((rand(n), rand(n), rand(n)))
    ax = ax.projectedPerpTo(v).norm
    ang = np.random.uniform(0, 2 * math.pi, n)
    vrot = v.rotated(ax, ang)
    checkAng, _ = v.anticlockAngleWith(vrot, ax)
    checkAng = checkAng % (2 * math.pi)
    assert np.allclose(ang, checkAng)
    v1 = Vec3((rand(n), rand(n), rand(n))).norm
    v2 = Vec3((rand(n), rand(n), rand(n))).norm
    ang, ax = v1.anticlockAngleWith(v2)
    v3 = v1.rotated(ax, ang).norm
    assert (v3.allClose(v2))


def test_Mat3Multiplication():
    m1 = Mat3((2, 4, 0), (1, 2, 6), (9, -1, 2))
    m2 = Mat3((-2, 2, 3), (5, -6, -1), (1, 1, 5))
    m1xm2 = Mat3((16, -20, 2), (14, -4, 31), (-21, 26, 38))
    assert m1 * m2 == m1xm2


def test_Vec3RotateByMat3():
    mi = Mat3.fromRotationParams(Vec3.i(), math.radians(90))
    mj = Mat3.fromRotationParams(Vec3.j(), math.radians(90))
    mk = Mat3.fromRotationParams(Vec3.k(), math.radians(90))
    atol = 1e-8
    assert (mj * Vec3.i()).allClose(-Vec3.k(), rtol=0, atol=atol)
    assert (mk * Vec3.i()).allClose(Vec3.j(), rtol=0, atol=atol)
    assert (mi * Vec3.j()).allClose(Vec3.k(), rtol=0, atol=atol)
    assert (mk * Vec3.j()).allClose(-Vec3.i(), rtol=0, atol=atol)
    assert (mi * Vec3.k()).allClose(-Vec3.j(), rtol=0, atol=atol)
    assert (mj * Vec3.k()).allClose(Vec3.i(), rtol=0, atol=atol)
    n = 100000
    v1 = Vec3((rand(n), rand(n), rand(n)))
    ax = Vec3((rand(n), rand(n), rand(n))).norm
    ang = np.random.normal(0, 200, n)
    v2truth = v1.rotated(ax, ang)
    v2test = Mat3.fromRotationParams(ax, ang) * v1
    assert v2test.allClose(v2truth, 1e-5, 0)


def test_Vec3RotateByQuaternion():
    qi = Quaternion.fromRotationParams(Vec3.i(), math.radians(90))
    qj = Quaternion.fromRotationParams(Vec3.j(), math.radians(90))
    qk = Quaternion.fromRotationParams(Vec3.k(), math.radians(90))
    atol = 1e-8
    assert Vec3.i().rotatedByQuaternion(qj).allClose(-Vec3.k(), rtol=0, atol=atol)
    assert Vec3.i().rotatedByQuaternion(qk).allClose(Vec3.j(), rtol=0, atol=atol)
    assert Vec3.j().rotatedByQuaternion(qi).allClose(Vec3.k(), rtol=0, atol=atol)
    assert Vec3.j().rotatedByQuaternion(qk).allClose(-Vec3.i(), rtol=0, atol=atol)
    assert Vec3.k().rotatedByQuaternion(qi).allClose(-Vec3.j(), rtol=0, atol=atol)
    assert Vec3.k().rotatedByQuaternion(qj).allClose(Vec3.i(), rtol=0, atol=atol)
    n = 10000
    v1 = Vec3((rand(n), rand(n), rand(n)))
    ax = Vec3((rand(n), rand(n), rand(n))).norm
    ang = np.random.normal(0, 200, n)
    v2truth = v1.rotated(ax, ang)
    v2test = v1.rotatedByQuaternion(Quaternion.fromRotationParams(ax, ang))
    assert v2test.allClose(v2truth, 1e-5, 0)
    v1 = Vec3((rand(n), rand(n), rand(n)))
    v2truth = Vec3((rand(n), rand(n), rand(n)))
    v2truth = v2truth.norm * v1.length
    q = v1.quaternionTo(v2truth)
    v2test = v1.rotatedByQuaternion(q)
    assert v2test.allClose(v2truth, 1e-5, 0)


def test_Mat3QuaternionRotations():
    n = 10000
    v1 = Vec3((rand(n), rand(n), rand(n)))
    v2 = Vec3((rand(n), rand(n), rand(n)))
    q1 = v1.quaternionTo(v2)
    m1 = Mat3.fromQuaternion(q1)
    v3 = m1 * v1
    assert v2.norm.allClose(v3.norm, 1e-6, 0)
    assert np.allclose(v2.angleWith(v3), 0, 0, 1e-7)
    v4 = Vec3((rand(n), rand(n), rand(n)))
    q2 = v2.quaternionTo(v4)
    q3 = q2 * q1
    m3 = Mat3.fromQuaternion(q3)
    v5 = m3 * v1
    v6 = v1.rotatedByQuaternion(q3)
    assert v4.norm.allClose(v5.norm, 1e-6, 0)
    assert np.allclose(v4.angleWith(v5), 0, 0, 1e-7)
    assert v6.allClose(v5, 1e-6, 0)
