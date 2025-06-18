# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module containing classes for defining renderable objects for use in image simulation."""

import numpy as np
import open3d
from coremaths.geometry import Geometry, Polygon
from coremaths.vector import Vec2, Vec3
from coremaths.frame import Frame
from coremaths.ray import Ray
from radiometry.reflectance_funcs import BRDF, TexturedBRDF as tBRDF, SpectralBRDF as sBRDF
from rendering.meshes import Mesh
from rendering import textures as tx
from rendering.lights import Light
from cameras.cameras import Camera
import warnings
from typing import Dict, List, Optional, Tuple, Union

_fnp = Union[float, np.ndarray]
_brdf = Union[BRDF, tBRDF, sBRDF]
_brdfs = Union[BRDF, tBRDF, sBRDF, List[BRDF], List[tBRDF], List[sBRDF]]


class RenderableObject:
    """Base class for an object which can be rendered using the renderer module

    Do not use instances of this class - only use instances of its subclasses"""
    def __init__(self, brdf: _brdf = None, tex: tx.textureType = None):
        """ Initialises a new renderable object base class.

        :param brdf: optional BRDF describing the reflection properties of the object's surface. Can be a standard BRDF,
            a textured BRDF or a spectral BRDF.
        :param tex: optional surface appearance texture
        """
        self._brdf = brdf
        self._texture = tex
        if type(brdf) is tBRDF and brdf.isTextured and tex is not None:
            message = ("All textures (i.e. its texture and any textured BRDF parameters) associated with a single "
                       "renderable object must be of the same texture type. If the renderable's texture must be of a "
                       "different texture type to textured brdf parameters, use two different renderable objects - one "
                       "with the textured brdf, and the other with the texture.")
            assert tx.textureIsCompatibleWithType(self._brdf.textureType, tex), message

    @staticmethod
    def renderablePrimitive(geometry: Geometry, brdf: Union[BRDF, tBRDF, sBRDF] = None, tex: tx.textureType = None):
        """ Returns a renderable object whose surface is described by the given geometric primitive.

        If physically-based radiometric rendering of this object is required, a BRDF or textured BRDF describing its
        surface reflectance must also be provided.

        A texture can also be provided to enable mapping the model's surface appearance directly from an image.

        :param geometry: the geometry describing the surface
        :param brdf: optional BRDF describing model's surface reflectance (required if conducting
                    radiometric rendering of the model)
        :param tex: optional surface appearance texture
        :return: the renderable surface
        """
        return RenderablePrimitive(geometry, brdf, tex)

    @staticmethod
    def renderableMesh(mesh: Mesh, brdf: Union[BRDF, tBRDF, sBRDF] = None, tex: tx.textureType = None):
        """ Returns a renderable object whose surface is described by the given surface mesh.

        If physically-based radiometric rendering of this object is required, a BRDF or textured BRDF describing its
        surface reflectance must also be provided.

        A texture can also be provided to enable mapping the model's surface appearance directly from an image.

        :param mesh: mesh describing the model's surface shape
        :param brdf: optional BRDF describing model's surface reflectance (required if conducting
                    radiometric rendering of the model)
        :param tex: optional surface appearance texture
        :return: the renderable object
        """
        return RenderableMesh(mesh, brdf, tex)

    @property
    def physicallyRenderable(self) -> bool:
        """Whether this renderable object has physical reflectance data, which is required for physical rendering"""
        return self._brdf is not None

    @property
    def texture(self) -> Optional[tx.textureType]:
        """The object's surface appearance texture, if any"""
        return self._texture

    @property
    def textureType(self):
        """The type of texture that this renderable object's associated textures (i.e. any textured brdf parameters,
         or its appearance texture) have (all textures associated with a renderable object must be of the same texture
         type)"""
        if self._texture is not None:
            return type(self._texture)
        if self._brdf is not None and type(self._brdf) is tBRDF:
            return self._brdf.textureType
        if self._brdf is not None and type(self._brdf) is sBRDF:
            return self._brdf.textureType
        return None

    @property
    def isPlanetocentricType(self):
        """If the texture type of this object is planetocentric (meaning its texture(s) are single planetocentric
        textures or compound planetocentric textures), this returns True."""
        return self.textureType is tx.TexturePlanetocentric or self.textureType is tx.CompoundTexturePlanetocentric

    def intersect(self, ray: Ray, max_dist: float = 1e15) -> Dict[str, np.ndarray]:
        """ Calculates and returns the depth and position of the first positive intersection of the given ray with this
        renderable object, if any.

        The returned result is a dictionary containing the hit depths ("t_hit"), hit surface normals
        ("primitive_normals") and hit geometry uv coordinates ("primitive_uvs"). If this renderable object's underlying
        surface is a trimesh, the returned dictionary will also contain hit triangle IDs ("primitive_ids") and geometry
        IDs ("geometry_ids").

        :param ray: The ray(s) to test for intersection with this renderable (must be in world coordinate frame)
        :param max_dist: Maximum distance of intersection. Any intersections beyond this distance are discarded.
        :return: Dictionary of the ray intersection result
        """
        message = "function not implemented - use a subclass of RenderableObject with implemented intersected function"
        warnings.warn(message)
        return {}

    def pIntersection(self, intersection: Dict[str, np.ndarray]) -> Optional[Vec3]:
        """ For a given result of an intersection with this renderable, this function returns the position, in world
        coordinates, of the intersection (using this function is more accurate than extrapolating position from
        intersection depth).

        :param intersection: The intersection result.
        :return: The intersection point (world frame)
        """
        message = "function not implemented - use subclass of RenderableObject with implemented pIntersection function"
        warnings.warn(message)
        return None

    def textureCoord(self, intersection: Dict[str, np.ndarray]) -> Optional[Union[Tuple[_fnp, _fnp], Vec3]]:
        """ For the given intersection result (where the intersection was performed on this object), this
        function returns the corresponding texture coordinate on the object's associated textures (i.e. textured BRDF
        parameters and/or texture map).

        If the object's associated textures (which must all be of the same type) are planetocentric textures, the
        returned coordinate will be a Vec3, otherwise it will be a uv tuple.

        :param intersection: dictionary of intersection result (as returned by this object's intersect(...) function)
        :return: the coordinate for mapping values from the texture, or None if this object has no texture
        """
        message = "function not implemented - use subclass of RenderableObject with implemented textureCoord function"
        warnings.warn(message)
        return None

    def textureValue(self, intersection: Dict[str, np.ndarray], rgb_channel=2) -> Optional[np.ndarray]:
        """ For the given intersection result (where the intersection was performed on this object), this
        function returns the surface appearance at the intersection point(s) mapped directly from the object's texture
        (if it has one).

        :param intersection: dictionary of intersection result (as returned by this object's intersect(...) function)
        :param rgb_channel: If the texture is an rgb texture (i.e. has 3 channels), rgb_channel dictates which channel
            the texture value will be taken from. Set rgb_channel to 1, 2 or 3 to use the texture's red, green or blue
            channel respectively. If the texture is not rgb (i.e. has a single channel), the value of rgb_channel is
            ignored.
        :return: the surface appearance at the intersection(s), or None if this object has no texture
        """
        if self._texture is None:
            # there is no texture assigned to this object
            return np.full_like(intersection['t_hit'], 0, dtype=float)
        coord = self.textureCoord(intersection)
        if coord is None:
            return None
        if type(coord) is Vec3:
            # the texture coord is a vector, which means the renderable mesh has an associated planetocentric texture
            return self._texture.valueFromXYZ(coord, rgb_channel=rgb_channel)
        # the texture coord is a uv tuple
        return self._texture.valueFromUV(coord[0], coord[1], rgb_channel=rgb_channel)

    def brdf(self, intersection: Dict[str, np.ndarray], w: float = None) -> Optional[BRDF]:
        """ For the given intersection result (where the intersection was performed on this object), this
        function returns the BRDF describing the surface's reflectance at the intersection point(s). The returned BRDF
        can then be used to calculate the reflected radiance at the intersection point(s).

        If this object has no BRDF assigned, this function returns None.

        :param intersection: dictionary of intersection result (as returned by this object's intersect(...) function)
        :param w: the wavelength [nm] at which the BRDF is desired (required only if the surface has a
                spectrally-dependent BRDF, otherwise leave as None)
        :return: surface's BRDF at the intersection point(s) (and given wavelength if relevant)
        """
        if self._brdf is None:
            # no BRDF assigned to this model
            return None
        if type(self._brdf) is sBRDF:
            # the brdf assigned to this model is wavelength dependent
            brdf = self._brdf.atWavelength(w)
            if brdf is None:
                if w is None:
                    msg = ("spectral brdf has no default brdf, so a wavelength matching one of its brdfs must be "
                           "passed here")
                    raise ValueError(msg)
                raise ValueError("spectral brdf has no brdf for wavelength {} nm".format(w))
        else:
            # the brdf assigned to this model is independent of wavelength
            brdf = self._brdf

        if type(brdf) is not tBRDF:
            # the brdf assigned to this model is spatially-uniform, so has no associated texture(s)
            return brdf
        else:
            # the BRDF assigned to this model is a textured BRDF (meaning some/all parameters are given by a texture)
            coord = self.textureCoord(intersection)
            return brdf.brdf(coord)

    def brdfEvaluated(self, intersection: Dict[str, np.ndarray], n: Vec3, ls: Vec3, v: Vec3, w: float = None) -> _fnp:
        """ For the given ray-object intersection result (where the intersection was performed on this object), this
        function calculates and returns the value of the BRDF (i.e. the radiance-over-irradiance ratio) at the
        intersection point(s) for the given reflection geometry.

        :param intersection: dictionary of intersection result (as returned by this object's intersect(...) function)
        :param n: the surface normal(s) at the intersection point(s)
        :param ls: normalised vector pointing from intersection point(s) to the light source
        :param v: normalised vector pointing from the intersection point(s) to the viewer
        :param w: the wavelength [nm] at which the BRDF is desired (required only if the surface has a
                spectrally-dependent BRDF, otherwise leave as None)
        :return: the evaluated BRDF [sr^-1]
        """
        return self.brdf(intersection, w).evaluate(n, ls, v)


class RenderablePrimitive(RenderableObject):
    """ Class for renderable objects whose surfaces are described by simple geometric primitives"""
    def __init__(self, geometry: Geometry, brdf: _brdf = None, tex: tx.textureType = None):
        """ Initialises a new renderable primitive object

        :param geometry: the geometric primitive defining its surface.
        :param brdf: optional BRDF describing the surface's reflection properties.
        :param tex: optional surface appearance texture.
        """
        super().__init__(brdf, tex)
        self.geometry = geometry

    @property
    def frame(self):
        """The geometry's frame, defining its position and orientation in 3d space"""
        return self.geometry.frame

    @frame.setter
    def frame(self, new_frame):
        """Sets a new frame defining this renderable surface's geometry's position and orientation"""
        self.geometry.frame = new_frame

    def intersect(self, ray: Ray, max_dist: float = 1e15, shift: Vec3 = None) -> Dict[str, np.ndarray]:
        """ Calculates and returns the depth and position of the first positive intersection of the given ray with this
        renderable mesh, if any.

        The returned result is a dictionary containing the hit depths ("t_hit"), hit surface normals
        ("primitive_normals"), hit primitive uv coordinates ("primitive_uvs") and hit geometry IDs ("geometry_ids").

        :param ray: The ray(s) to test for intersection with this renderable (must be in world coordinate frame)
        :param max_dist: Maximum distance of intersection. Any intersections beyond this distance are discarded.
        :param shift: Vector by which the geometry is shifted before performing intersection test (geometry is shifted
            back after intersection test). Can be used to shift the geometry closer to the origin (if the input rays
            have been shifted by the same vector) to improve ray tracing accuracy by reducing floating point errors.
        :return: Dictionary of the ray intersection result
        """
        return self.geometry.intersect(ray, max_dist=max_dist, shift=shift)

    def pIntersection(self, intersection: Dict[str, np.ndarray]) -> Optional[Vec3]:
        if intersection:
            uv = intersection['primitive_uvs']
            uv = Vec2.fromNumpyArray(uv)
            return self.geometry.pointFromUV(uv.x, uv.y)
        return None

    def textureCoord(self, intersection: Dict[str, np.ndarray]) -> Optional[Union[Tuple[_fnp, _fnp], Vec3]]:
        if self.textureType is None:
            # there are no textures associated with this object
            return None
        uv = intersection['primitive_uvs']
        uv = Vec2.fromNumpyArray(uv)
        if self.isPlanetocentricType:
            # object's assigned texture is planetocentric
            pHit = self.geometry.pointFromUV(uv.x, uv.y)
            return pHit
        else:
            # object's assigned texture is not planetocentric, so instead use primitive uv coord as texture uv coord
            return uv.x, uv.y


class RenderableMesh(RenderableObject):
    """Class for a renderable objects whose surfaces are described by triangle meshes."""

    def __init__(self, mesh: Mesh, brdf: Union[BRDF, tBRDF] = None, tex: tx.textureType = None):
        """ Initialises a new renderable object with the given surface mesh.

        If physically-based radiometric rendering of this object is required, a BRDF or textured BRDF describing its
        surface reflectance must also be provided.

        A texture can also be provided to enable mapping the model's surface appearance directly from an image.

        :param mesh: mesh describing the model's surface shape
        :param brdf: optional BRDF describing model's surface reflectance (required if conducting
                    radiometric rendering of the model)
        :param tex: optional surface appearance texture
        """
        super().__init__(brdf, tex)
        self._mesh = mesh

    @property
    def mesh(self) -> Mesh:
        """The model's surface mesh"""
        return self._mesh

    def quickView(self, fov: Tuple[float, float], dc: int, dr: int, r: float):
        """ Returns a pinhole camera which is looking at the centre of the renderable mesh and can be used for rendering
         visualisations of it.

        :param fov: field of view of the camera [degrees]
        :param dc: number of columns on camera's detector
        :param dr: number of rows on camera's detector
        :param r: distance from camera to centre of mesh [m]
        :return: the camera
        """
        meshCentre = self._mesh.meanVert
        axis = meshCentre.norm
        camera = Camera.pinhole(fov, dc, dr)
        camera.frame = Frame.withW(-axis, origin=meshCentre + r * axis)
        return camera

    def textureCoord(self, intersection: Dict[str, np.ndarray]) -> Optional[Union[Tuple[_fnp, _fnp], Vec3]]:
        """ For the given intersection result (where the intersection was performed on this model's mesh), this function
         returns the corresponding texture coordinate on the model's texture map (if it has one).

         If the model's assigned texture is a planetocentric texture, the returned coordinate will be a Vec3, otherwise
         it will be a uv tuple.

        :param intersection: dictionary of intersection result (as returned by this model's mesh's intersect() function)
        :return: the coordinate for mapping values from the texture, or None if this model has no texture
        """
        if self.textureType is None:
            # there are no textures associated with this renderable mesh
            return None
        if self._mesh.hasVertexTextureMapping:
            # the renderable mesh has texture coordinates assigned to its vertices for mapping to a texture
            texCoordArray = self._mesh.textureCoordArray
            triID = intersection['primitive_ids']
            triIsNan = np.isnan(triID)
            triID_int = triID.astype(int)
            triID_int[triIsNan] = 0
            baseSlice = (slice(None),) * triID.ndim
            texCoordIndex = self._mesh.triTexIndices[triID_int][baseSlice + (0,)].astype(float)
            texCoordIndex[triIsNan] = np.nan
            texIsNan = np.isnan(texCoordIndex)
            texCoordIndex_int = texCoordIndex.astype(int)
            texCoordIndex_int[texIsNan] = 0
            u = texCoordArray[:, 0][texCoordIndex_int]
            v = texCoordArray[:, 1][texCoordIndex_int]
            u[texIsNan] = np.nan
            v[texIsNan] = np.nan
            return u, v
        if self.isPlanetocentricType:
            # the renderable mesh has an associated planetocentric texture
            primID = intersection['primitive_ids']
            primUV = intersection['primitive_uvs']
            baseSlice = (slice(None),) * primID.ndim
            pHit = self._mesh.coordFromTriUV(primID, primUV[baseSlice + (0,)], primUV[baseSlice + (1,)])
            return pHit
        if self._mesh.isGridMesh:
            # the renderable mesh is of grid type, so its grid uv coords can be used to map to a texture
            primID = intersection['primitive_ids']
            primUV = intersection['primitive_uvs']
            baseSlice = (slice(None),) * primID.ndim
            u, v = self._mesh.triUVToGridUV(primID, primUV[baseSlice + (0,)], primUV[baseSlice + (1,)])
            return u, v
        # a texture has been assigned to the renderable mesh, but mapping to the texture is not defined or inferable
        return None

    def intersect(self, ray: 'Ray', max_dist: float = 1e15):
        """ Performs an intersection test of the given ray(s) with this model's mesh and returns the result.

        The returned result is a dictionary containing the hit depths ("t_hit"), hit triangle IDs ("primitive_ids"),
        hit triangle surface normals ("primitive_normals"), hit triangle uv coordinates ("primitive_uvs") and
        geometry id ("geometry_ids").

        The ray mesh intersection is performed using open3D.

        :param ray: The ray or rays (represent multiple rays by using numpy arrays for the ray's vectors' elements)
        :param max_dist: Maximum distance of intersection. Any intersections beyond this distance are discarded.
        :return: The ray intersection result.
        """
        return self._mesh.intersect(ray, max_dist)

    def pIntersection(self, intersection: Dict[str, np.ndarray]) -> Optional[Vec3]:
        primID = intersection['primitive_ids']
        primUV = intersection['primitive_uvs']
        uv = Vec2.fromNumpyArray(primUV)
        return self._mesh.coordFromTriUV(primID, uv.x, uv.y)


class RenderableScene:
    """Class for a scene containing renderable objects, of which simulated images can be rendered."""
    def __init__(self, renderables: List[RenderableObject], light: Light = None, shift: Union['Vec3', bool] = True):
        """ Initialises a new renderable scene containing the given renderable objects and light.

        :param renderables: list of renderable objects that the scene contains
        :param light: the light source in the scene
        :param shift: If set to True, the scene will be shifted prior to ray tracing to centre it on the coordinate
            origin. If a vector is passed for this argument, the scene will be shifted by that vector prior to ray
            tracing. The scene is returned to its original position after ray tracing. Shifting of the scene can
            significantly improve ray tracing accuracy when all objects within it are located far from the origin.
        """
        self._primitives: List[RenderablePrimitive] = []
        self._meshes: List[RenderableMesh] = []
        for renderable in renderables:
            if isinstance(renderable, RenderablePrimitive):
                self._primitives += [renderable]
            elif isinstance(renderable, RenderableMesh):
                self._meshes += [renderable]
        self._light = light

        self._sceneO3D = None
        if type(shift) is Vec3:
            self._renderingShift: 'Vec3' = shift
        elif shift:
            self._renderingShift = -self._approxCentreOfRenderables
        else:
            self._renderingShift = None
        for mesh in self._meshes:
            mesh.mesh.renderingShift = self._renderingShift

    @classmethod
    def polygons(cls, polys: List[Polygon], brdfs: _brdfs, light: Light, shift: Union['Vec3', bool] = True):
        """ Initialises a renderable scene containing the given polygons, whose reflection properties are defined by the
         given brdf or brdfs.

         Note: this is well suited to a scene containing only a small number of polygons, and should not be used for
         scenes with surface meshes.

        :param polys: List of polygons in the scene.
        :param brdfs: Either a single BRDF applying to all given polygons, or a list of one BRDF per polygon.
        :param light: The scene's illumination source.
        :param shift: If set to True, the scene will be shifted prior to ray tracing to centre it on the coordinate
            origin. If a vector is passed for this argument, the scene will be shifted by that vector prior to ray
            tracing. The scene is returned to its original position after ray tracing. Shifting of the scene can
            significantly improve ray tracing accuracy when all objects within it are located far from the origin.
        :return: the scene.
        """
        renderables = []
        for n, poly in enumerate(polys):
            if type(brdfs) is list:
                brdf = brdfs[n]
            else:
                brdf = brdfs
            renderables += [RenderablePrimitive(poly, brdf)]
        return cls(renderables, light, shift=shift)

    @property
    def physicallyRenderable(self):
        """Whether all the objects within this scene have physical reflectance data, which is required for physical
        rendering of this scene"""
        if self._light is None:
            return False
        for renderable in self._primitives:
            if not renderable.physicallyRenderable:
                return False
        for renderable in self._meshes:
            if not renderable.physicallyRenderable:
                return False
        return True

    @property
    def _approxCentreOfRenderables(self) -> Optional[Vec3]:
        """Returns the approximate geometric centre of all this scene's renderable objects by returning the mean of all
        of their centres"""
        count = len(self._primitives) + len(self._meshes)
        if count == 0:
            return None
        renderableCentre = Vec3.zero()
        for primitive in self._primitives:
            renderableCentre += primitive.frame.origin
        for mesh in self._meshes:
            renderableCentre += mesh.mesh.meanVert
        return renderableCentre / count

    @property
    def light(self):
        """The scene's light source (or None if the scene has no light source)."""
        return self._light

    @light.setter
    def light(self, new_light: Light):
        """Sets the scene's light source."""
        self._light = new_light

    def intersect(self, ray: Ray, max_dist: float = 1e15) -> Dict[str, np.ndarray]:
        """ Performs an intersection test of the given ray(s) with each of this scene's objects in turn, and returns the
        result of the nearest intersection (if any) for each ray.

        The returned result is a dictionary containing the hit depths (key="t_hit"), hit geometry IDs
        (key="geometry_ids"), hit primitive triangle IDs (only if the object is a mesh, key="primitive_ids"), hit
        surface normals (key="primitive_normals") and hit primitive uv coordinates (key="primitive_uvs").
        Geometry ids >= 0 are mesh objects, <0 are geometric primitives.

        :param ray: The ray or rays (represent multiple rays by using numpy arrays for the ray's vectors' elements)
        :param max_dist: Maximum distance of intersection. Any intersections beyond this distance are discarded.
        :return: The ray intersection result.
        """
        # translate ray(s) to mesh's rendering-shifted frame (--> higher accuracy by reducing floating point errors):
        if self._renderingShift is not None:
            ray = Ray(ray.origin + self._renderingShift, ray.d)

        if self._sceneO3D is None:
            self._sceneO3D = open3d.t.geometry.RaycastingScene()
            for mesh in self._meshes:
                self._sceneO3D.add_triangles(mesh.mesh.meshO3D)

        if ray.isNumpyType:
            # ray object is numpy type, representing many individual rays
            shape = ray.numpyShape
            notNan = ~(ray.origin.isNan + ray.d.isNan)
            noNanRay = ray.numpyMasked(notNan)
            rayO3D = np.empty((np.prod(noNanRay.numpyShape), 6), dtype=np.dtype('float32'))
            for i in range(3):
                try:
                    rayO3D[:, i] = ray.origin[i][notNan]
                except IndexError:
                    rayO3D[:, i] = ray.origin[i]
                except TypeError:
                    rayO3D[:, i] = ray.origin[i]
                try:
                    rayO3D[:, i + 3] = ray.d[i][notNan]
                except IndexError:
                    rayO3D[:, i + 3] = ray.d[i]
                except TypeError:
                    rayO3D[:, i + 3] = ray.d[i]

            ans = self._sceneO3D.cast_rays(rayO3D)
            hitDepths = np.full(shape, np.nan, dtype=float)
            hitDepths[notNan] = ans['t_hit'].numpy()
            outRange = hitDepths > max_dist
            hitDepths[outRange] = np.nan
            geomIDs = np.full(shape, np.nan, dtype=float)
            geomIDs[notNan] = ans['geometry_ids'].numpy()
            geomIDs[outRange] = np.nan
            primIDs = np.full(shape, np.nan, dtype=float)
            primIDs[notNan] = ans['primitive_ids'].numpy()
            primIDs[outRange] = np.nan
            norms = np.full(shape + (3,), np.nan, dtype=float)
            norms[notNan] = ans['primitive_normals'].numpy()
            uvs = np.full(shape + (2,), np.nan, dtype=float)
            uvs[notNan] = ans['primitive_uvs'].numpy()

            result = {'t_hit': hitDepths, 'geometry_ids': geomIDs, 'primitive_ids': primIDs, 'primitive_normals': norms,
                      'primitive_uvs': uvs}

            for geomID, primitive in enumerate(self._primitives):
                ret = primitive.intersect(ray, max_dist=max_dist, shift=self._renderingShift)
                nearest = ((ret['t_hit'] < result['t_hit']) + np.isnan(result['t_hit'])) * ~np.isnan(ret['t_hit'])
                result['t_hit'] = np.where(nearest, ret['t_hit'], result['t_hit'])
                normals = np.copy(result['primitive_normals'])
                normals[nearest] = ret['primitive_normals'][nearest]
                result['primitive_normals'] = normals
                uvs = np.copy(result['primitive_uvs'])
                uvs[nearest] = ret['primitive_uvs'][nearest]
                result['primitive_uvs'] = uvs
                result['geometry_ids'][nearest] = -geomID - 1

            return result
        else:
            # ray object represents a single ray
            rayO3D = np.array([ray.origin.tuple + ray.d.tuple], dtype=np.dtype('float32'))

            ans = self._sceneO3D.cast_rays(rayO3D)
            hitDepths = ans['t_hit'].numpy()
            geomIDs = ans['geometry_ids'].numpy()
            primIDs = ans['primitive_ids'].numpy()
            norms = ans['primitive_normals'].numpy()
            uvs = ans['primitive_uvs'].numpy()

            result = {'t_hit': hitDepths, 'geometry_ids': geomIDs, 'primitive_ids': primIDs,
                      'primitive_normals': norms,
                      'primitive_uvs': uvs}

            for geomID, primitive in enumerate(self._primitives):
                ret = primitive.intersect(ray, max_dist=max_dist)
                ret['geometry_ids'] = np.array((-geomID - 1,))
                if ret['t_hit'] < result['t_hit']:
                    result = ret
            return result

    def pIntersection(self, intersection: Dict[str, np.ndarray]) -> Optional[Vec3]:
        """ For a given result of an intersection with this scene, this function returns the position, in world
        coordinates, of the intersection (using this function is more accurate than extrapolating position from
        intersection depth).

        :param intersection: The intersection result.
        :return: The intersection point (world frame)
        """
        ret = np.full_like(intersection['primitive_normals'], np.nan, dtype=float)
        for meshID, mesh in enumerate(self._meshes):
            match = intersection['geometry_ids'] == meshID
            if np.any(match):
                ret[match] = mesh.pIntersection(intersection).asNumpyArray[match]
        for geomID, primitive in enumerate(self._primitives):
            # geometry ids for primitive geometry objects are equal to (-1 - n)
            # where n is the geometry's index in the self._primitives list
            match = -1 - intersection['geometry_ids'] == geomID
            if np.any(match):
                ret[match] = primitive.pIntersection(intersection).asNumpyArray[match]
        return Vec3.fromNumpyArray(ret)

    def textureValue(self, intersection: Dict[str, np.ndarray], rgb_channel=2) -> Optional[np.ndarray]:
        """ For the given intersection result (where the intersection was performed on this scene), this
        function returns the surface appearance at the intersection point(s) mapped directly from the scene objects'
        textures (if they have them).

        :param intersection: dictionary of intersection result (as returned by this scene's intersect(...) function)
        :param rgb_channel: If the texture is an rgb texture (i.e. has 3 channels), rgb_channel dictates which channel
            the texture value will be taken from. Set rgb_channel to 1, 2 or 3 to use the texture's red, green or blue
            channel respectively. If the texture is not rgb (i.e. has a single channel), the value of rgb_channel is
            ignored.
        :return: the surface appearance at the intersection(s), or None if/where the intersected surface has no texture
        """
        ret = np.full(intersection['t_hit'].shape, np.nan, dtype=float)
        for meshID, mesh in enumerate(self._meshes):
            match = intersection['geometry_ids'] == meshID
            if np.any(match):
                textureValue = mesh.textureValue(intersection, rgb_channel=rgb_channel)
                if textureValue is not None:
                    textureValue = textureValue[match]
                    ret[match] = textureValue
                else:
                    ret[match] = 0
        for geomID, primitive in enumerate(self._primitives):
            match = -1 - intersection['geometry_ids'] == geomID
            if np.any(match):
                textureValue = primitive.textureValue(intersection, rgb_channel=rgb_channel)
                if textureValue is not None:
                    textureValue = textureValue[match]
                    ret[match] = textureValue
                else:
                    ret[match] = 0
        return ret

    def brdfEvaluated(self, intersection: Dict[str, np.ndarray], n: Vec3, ls: Vec3, v: Vec3, w: float = None) -> _fnp:
        """ For the given ray-scene intersection result (where the intersection was performed on this scene), this
        function calculates and returns the value of the BRDF (i.e. the radiance-over-irradiance ratio) at the
        intersection point(s) for the given reflection geometry (and given wavelength if relevant).

        :param intersection: dictionary of intersection result (as returned by this scene's intersect(...) function)
        :param n: the surface normal(s) at the intersection point(s)
        :param ls: normalised vector pointing from intersection point(s) to the light source
        :param v: normalised vector pointing from the intersection point(s) to the viewer
        :param w: the wavelength [nm] at which the BRDF is desired (required only if the scene contains
                spectrally-dependent BRDFs, otherwise leave as None)
        :return: the evaluated BRDF [sr^-1]
        """
        retShape = ls.numpyShape
        ret = np.full(retShape, np.nan, dtype=float)
        for meshID, mesh in enumerate(self._meshes):
            seesMesh: Union[np.ndarray, bool] = intersection['geometry_ids'] == meshID
            if seesMesh.shape != retShape:
                shapeDiff = retShape[:-len(seesMesh.shape)]
                for axisLength in reversed(shapeDiff):
                    seesMesh = np.repeat(seesMesh[None], axisLength, axis=0)
            meshRet = mesh.brdf(intersection, w).evaluate(n, ls, v)
            ret[seesMesh] = meshRet[seesMesh]
        for primID, primitive in enumerate(self._primitives):
            primID = -primID - 1
            seesPrim: Union[np.ndarray, bool] = intersection['geometry_ids'] == primID
            if seesPrim.shape != retShape:
                shapeDiff = retShape[:-len(seesPrim.shape)]
                for axisLength in reversed(shapeDiff):
                    seesPrim = np.repeat(seesPrim[None], axisLength, axis=0)
            primitiveRet = primitive.brdf(intersection, w).evaluate(n, ls, v)
            ret[seesPrim] = primitiveRet[seesPrim]
        return ret
