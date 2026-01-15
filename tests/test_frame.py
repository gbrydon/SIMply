import math
import numpy as np
from coremaths.vector import Vec3, Mat3
from coremaths.frame import Frame


def assertFrameAxesUnitary(frame: 'Frame'):
    assert np.allclose(frame.u.length, 1, 0, 1e-7)
    assert np.allclose(frame.v.length, 1, 0, 1e-7)
    assert np.allclose(frame.w.length, 1, 0, 1e-7)


def assertFrameHandednessCorrect(frame: 'Frame'):
    u = frame.v.cross(frame.w)
    assert u.allClose(frame.u)
    v = frame.w.cross(frame.u)
    assert v.allClose(frame.v)
    w = frame.u.cross(frame.v)
    assert w.allClose(frame.w)


def assertFrameAxesOrthogonal(frame: 'Frame'):
    assert np.allclose(frame.u.angleWith(frame.v), 0.5 * np.pi, 0, math.radians(1e-7))
    assert np.allclose(frame.u.angleWith(frame.w), 0.5 * np.pi, 0, math.radians(1e-7))
    assert np.allclose(frame.v.angleWith(frame.w), 0.5 * np.pi, 0, math.radians(1e-7))


def test_worldFrame():
    f = Frame.world()
    assert f.u == Vec3.i()
    assert f.v == Vec3.j()
    assert f.w == Vec3.k()


def test_framesFromVectors():
    def checkFrameConstruction(frame: 'Frame'):
        assertFrameAxesUnitary(frame)
        assertFrameAxesOrthogonal(frame)
        assertFrameHandednessCorrect(frame)

    n = 10000
    vec1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    checkFrameConstruction(Frame.withU(vec1))
    checkFrameConstruction(Frame.withV(vec1))
    checkFrameConstruction(Frame.withW(vec1))
    vec2 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    checkFrameConstruction(Frame.withU(vec1, v=vec2))
    checkFrameConstruction(Frame.withU(vec1, w=vec2))
    checkFrameConstruction(Frame.withV(vec1, u=vec2))
    checkFrameConstruction(Frame.withV(vec1, w=vec2))
    checkFrameConstruction(Frame.withW(vec1, u=vec2))
    checkFrameConstruction(Frame.withW(vec1, v=vec2))

    def checkBasesEqualWorld(frame):
        fu = Frame.world()
        assert fu.u == frame.u
        assert fu.v == frame.v
        assert fu.w == frame.w
    f = Frame.withU(Vec3.i(), v=Vec3.j())
    checkBasesEqualWorld(f)
    f = Frame.withU(Vec3.i(), w=Vec3.k())
    checkBasesEqualWorld(f)
    f = Frame.withV(Vec3.j(), u=Vec3.i())
    checkBasesEqualWorld(f)
    f = Frame.withV(Vec3.j(), w=Vec3.k())
    checkBasesEqualWorld(f)
    f = Frame.withW(Vec3.k(), u=Vec3.i())
    checkBasesEqualWorld(f)
    f = Frame.withW(Vec3.k(), v=Vec3.j())
    checkBasesEqualWorld(f)


def test_frameWorldConversions():
    n = 10000
    u1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n))).norm
    o1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    frame1 = Frame.withV(u1, origin=o1)
    p1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    pWorld1 = frame1.toWorld(p1)
    pWorld2 = Frame.world().fromFrame(p1, frame1)
    assert pWorld1.allClose(pWorld2)
    p2 = frame1.fromWorld(pWorld1)
    assert p1.allClose(p2)


def test_frameToFrameConversions():
    n = 100000
    u1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    u2 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    o1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    o2 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    frame1 = Frame.withU(u1.norm, origin=o1)
    frame2 = Frame.withU(u2.norm, origin=o2)
    p1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    p2 = frame1.toFrame(p1, frame2)
    p3 = frame2.fromFrame(p1, frame1)
    assert p2.allClose(p3)
    p4 = frame1.fromFrame(p2, frame2)
    assert p1.allClose(p4, rtol=1e-5, atol=0)


def test_frameToFrameRigidTransform():
    n = 100000
    u1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    u2 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    o1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    o2 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    frame1 = Frame.withU(u1.norm, origin=o1)
    frame2 = Frame.withU(u2.norm, origin=o2)
    R1, t1 = frame1.rigidTransformTo(frame2)
    R2, t2 = frame2.rigidTransformFrom(frame1)
    ptest = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    p1 = R1 * ptest + t1
    p2 = R2 * ptest + t2
    p1.allClose(p2)
    R3, t3 = frame1.rigidTransformFrom(frame2)
    p4 = R3 * p1 + t3
    ptest.allClose(p4)
    p5 = frame2.fromFrame(ptest, frame1)
    p1.allClose(p5)


def test_frameToFrameRotation():
    n = 100000
    u1 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    u2 = Vec3((np.random.normal(0, 100, n), np.random.normal(0, 100, n), np.random.normal(0, 100, n)))
    frame1 = Frame.withU(u1)
    frame2 = Frame.withU(u2)
    q = frame1.quaternionTo(frame2)
    m = Mat3.fromQuaternion(q)
    frame3 = frame1.rotatedInPlaceByMatrix(m)
    assert frame2.u.allClose(frame3.u)
    assert frame2.v.allClose(frame3.v)
    assert frame2.w.allClose(frame3.w)
