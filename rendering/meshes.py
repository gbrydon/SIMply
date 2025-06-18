# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module for working with triangle meshes."""

import numpy as np
import open3d
from coremaths.vector import Vec2, Vec3, Mat3
from coremaths.ray import Ray
from coremaths.frame import Frame
from typing import Dict, List, Optional, Tuple, Union

_fnp = Union[float, np.ndarray]
_inp = Union[int, np.ndarray]
_np = np.ndarray


class Mesh:
    """Class describing a triangle mesh (see e.g. https://en.wikipedia.org/wiki/Triangle_mesh).

    The mesh is defined by an array of vertices (points in 3D space) and an array of triangle facets (triplets of
    vertices, where each vertex is indicated by its index in the vertices array).
    """
    def __init__(self, vertices: _np, tris: _np, shift: Union['Vec3', bool] = True, frame: Frame = None):
        """ Initiates a new mesh defined by the given vertices and triangles.

        :param vertices: the vertices (3D points) that make up the mesh. For a mesh of n vertices, this is a numpy array
                    of shape (n, 3) containing the x, y, z coordinate of each vertex (in the mesh's coordinate frame).
        :param tris: the triangles making up the surface of the mesh. Each triangle is defined by the indices of its
                    three vertices in the vertices array. For a mesh of m triangles, this is a numpy array of shape
                    (m, 3). Note that the triangle indices start from 0, unlike in an .obj file where they start from 1.
        :param shift: If set to True, any ray tracing performed on this mesh will first shift the ray tracing scene to
                    centre the mesh on the coordinate origin. If set to False, no shifting is performed. If a vector is
                    passed for this argument, the scene will be shifted by the given vector before ray tracing. This
                    argument only has effect if ray tracing of this mesh is being performed. Shifting of the mesh
                    significantly improves the accuracy of the open3d ray tracing for meshes which are located far from
                    the origin (such as a small portion of a planetary surface).
        :param frame: the coordinate frame of the given vertex coordinates. Vertex coordinates are converted to the
            world frame during initialisation of the mesh. If set to None, the vertex coordinates are treated as already
            being in the world frame.
        """
        if frame is None:
            self._vertices = vertices
        else:
            self._vertices = frame.toWorld(Vec3.fromNumpyArray(vertices)).asNumpyArray
        self._tris = tris.astype(int)
        self._textureCoordArray = None
        self._triTexIndices = None
        self._meshO3D = None
        self._sceneO3D = None
        if type(shift) is Vec3:
            self._renderingShift: 'Vec3' = shift
        elif shift:
            self._renderingShift = -self.meanVert
        else:
            self._renderingShift = Vec3.zero()

        self._gridShape = None

    @staticmethod
    def vertsAndTrisFromPointGrid(x: _np, y: _np, z: _np) -> Tuple[_np, _np]:
        """Returns the vertices and triangles of a mesh formed from a regular 2D grid of 3D points (e.g. from a DTM).
        The vertices are returned as a list of shape [n, 3] corresponding to n (x,y,z) coordinates.
        Triangles are defined by the indices, in the vertices array, of each of their three vertices. These are returned
        as a list of shape [m, 3].

        :param x: the x coordinates of the points in the 2D regular grid of 3D points.
        :param y: the y coordinates of the points in the 2D regular grid of 3D points.
        :param z: the z coordinates of the points in the 2D regular grid of 3D points.
        :return: tuple containing the vertices and triangles arrays that define the mesh.
        """
        assert ((x.shape == y.shape) and (y.shape == z.shape)), "numpy arrays x, y and z must have the same shape"
        assert (len(x.shape) == 2), ("x, y and z must be two-dimensional numpy arrays, "
                                     "but have shape of {0}").format(x.shape, y.shape, z.shape)

        nr, nc = x.shape
        vertices = np.empty((x.size, 3), dtype='float64')
        vertices[:, 0] = x.ravel()
        vertices[:, 1] = y.ravel()
        vertices[:, 2] = z.ravel()
        flatIndex = np.arange(nr * nc).reshape((nr, nc))
        trisA = np.empty((nr - 1, nc - 1, 3), dtype=int)
        trisA[:, :, 0] = flatIndex[:-1, :-1]
        trisA[:, :, 1] = flatIndex[1:, :-1]
        trisA[:, :, 2] = flatIndex[:-1, 1:]
        trisB = np.empty((nr - 1, nc - 1, 3), dtype=int)
        trisB[:, :, 0] = flatIndex[1:, :-1]
        trisB[:, :, 1] = flatIndex[1:, 1:]
        trisB[:, :, 2] = flatIndex[:-1, 1:]
        trisA = trisA.reshape(((nr - 1) * (nc - 1), 3))
        trisB = trisB.reshape(((nr - 1) * (nc - 1), 3))
        tris = np.append(trisA, trisB, axis=0)
        nantris = np.any(np.any(np.isnan(vertices[tris]), axis=2), axis=1)
        tris = tris[~nantris]
        return vertices, tris

    @classmethod
    def loadFromVertsTris(cls, path: str):
        """ Returns a trimesh loaded from numpy arrays of the mesh's vertices and triangles at the given path.

        For a given path, verts and tris arrays will be loaded from path+"_verts.np" and path+"_tris.npy" respectively

        :param path: path from which to load trimesh
        :return: loaded trimesh
        """
        verts = np.load(path + '_verts.npy', allow_pickle=False)
        tris = np.load(path + '_tris.npy', allow_pickle=False)
        return cls(verts, tris)

    @classmethod
    def loadFromPointGrid(cls, path: str, dsf=1):
        """ Returns a trimesh loaded from a (MxNx3) numpy array of 3D points (i.e. a MxN grid of vertices) at the given
        path.

        :param path: path of the numpy array containing the point grid.
        :param dsf: down-sampling factor - integer factor by which the pointgrid will be downsampled when loaded
            (dsf=1 performs no down-sampling).
        :return: loaded trimesh
        """
        arr = np.load(path, allow_pickle=False)
        return cls.fromPointGrid(arr[::dsf, ::dsf, 0], arr[::dsf, ::dsf, 1], arr[::dsf, ::dsf, 2])

    @classmethod
    def fromPointGrid(cls, x: _np, y: _np, z: _np, shift: Union['Vec3', bool] = True, frame: Frame = None) -> 'Mesh':
        """Returns the mesh formed from a regular 2D grid of 3D points (e.g. from a DTM).

        :param x: the x coordinates of the points in the 2D regular grid of 3D points.
        :param y: the y coordinates of the points in the 2D regular grid of 3D points.
        :param z: the z coordinates of the points in the 2D regular grid of 3D points.
        :param shift: If set to True, any ray tracing performed on this mesh will first shift the ray tracing scene to
                    centre the mesh on the coordinate origin. If set to False, no shifting is performed. If a vector is
                    passed for this argument, the scene will be shifted by the given vector before ray tracing. This
                    argument only has effect if ray tracing of this mesh is being performed. Shifting of the mesh
                    significantly improves the accuracy of the open3d ray tracing for meshes which are located far from
                    the origin (such as a small portion of a planetary surface).
        :param frame: the coordinate frame of the given vertex coordinates. Vertex coordinates are converted to the
            world frame during initialisation of the mesh. If set to None, the vertex coordinates are treated as already
            being in the world frame.
        :return: the mesh formed from the point grid.
        """
        vertices, tris = Mesh.vertsAndTrisFromPointGrid(x, y, z)
        mesh = cls(vertices, tris, shift=shift, frame=frame)
        mesh._gridShape = x.shape
        return mesh

    @classmethod
    def fromVec3(cls, vertices: Vec3, tris: _np, shift: Union['Vec3', bool] = True, frame: Frame = None):
        """ Initialises a new mesh defined by the given vertices and triangles, where the vertices are provided as a
        numpy-type Vec3.

        :param vertices: The vertices of the mesh represented by a numpy-type Vec3
        :param tris: the triangles making up the surface of the mesh. Each triangle is defined by the indices of its
            three vertices in the vertices array. For a mesh of m triangles, this is a numpy array of shape
            (m, 3). Note that the triangle indices start from 0, unlike in an .obj file where they start from 1.
        :param shift: If set to True, any ray tracing performed on this mesh will first shift the ray tracing scene to
            centre the mesh on the coordinate origin. If set to False, no shifting is performed. If a vector is
            passed for this argument, the scene will be shifted by the given vector before ray tracing. This
            argument only has effect if ray tracing of this mesh is being performed. Shifting of the mesh
            significantly improves the accuracy of the open3d ray tracing for meshes which are located far from
            the origin (such as a small portion of a planetary surface).
        :param frame: the coordinate frame of the given vertex coordinates. Vertex coordinates are converted to the
            world frame during initialisation of the mesh. If set to None, the vertex coordinates are treated as already
            being in the world frame.
        :return: the mesh
        """
        v = np.empty((vertices.x.size, 3))
        v[:, 0] = vertices.x
        v[:, 1] = vertices.y
        v[:, 2] = vertices.z
        return cls(v, tris, shift=shift, frame=frame)

    @classmethod
    def loadFromOBJ(cls, path: str, shift: Union['Vec3', bool] = True, frame: Frame = None, sf: float = 1):
        """ Initialises a mesh from an .obj (wavefront) file.

        If the .obj file contains texture mapping, this
        is also loaded to the mesh and stored in the textureCoordArray and triTexIndices properties.

        :param path: The path of the .obj file.
        :param shift: If set to True, any ray tracing performed on this mesh will first shift the ray tracing scene to
            centre the mesh on the coordinate origin. If set to False, no shifting is performed. If a vector is
            passed for this argument, the scene will be shifted by the given vector before ray tracing. This
            argument only has effect if ray tracing of this mesh is being performed. Shifting of the mesh
            significantly improves the accuracy of the open3d ray tracing for meshes which are located far from
            the origin (such as a small portion of a planetary surface).
        :param frame: the coordinate frame of the given vertex coordinates. Vertex coordinates are converted to the
            world frame during initialisation of the mesh. If set to None, the vertex coordinates are treated as already
            being in the world frame.
        :param sf: scale factor by which the vertices' coordinate values in the obj file are multiplied (e.g. for
            converting from kilometre to metre).
        :return: the mesh.
        """
        vertices = []
        tris = []
        texCoords = []
        triVertTexIndices = []
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    mesh = cls(np.array(vertices), np.array(tris), shift=shift, frame=frame)
                    if texCoords and triVertTexIndices:
                        mesh.textureCoordArray = np.array(texCoords)
                        mesh.triTexIndices = np.array(triVertTexIndices)
                    return mesh
                line = line.split()
                if len(line) == 0:
                    continue
                if line[0] == 'v':
                    x = float(line[1]) * sf
                    y = float(line[2]) * sf
                    z = float(line[3]) * sf
                    vertices += [[x, y, z]]
                elif line[0] == 'f':
                    element1 = line[1].split('/')
                    element2 = line[2].split('/')
                    element3 = line[3].split('/')
                    v1 = int(element1[0]) - 1
                    v2 = int(element2[0]) - 1
                    v3 = int(element3[0]) - 1
                    tris += [[v1, v2, v3]]
                    if len(element1) > 1 and len(element2) > 1 and len(element3) > 1:
                        if element1[1] != '' and element2[1] != '' and element2[1] != '':
                            vti1 = int(element1[1]) - 1
                            vti2 = int(element2[1]) - 1
                            vti3 = int(element3[1]) - 1
                            triVertTexIndices += [[vti1, vti2, vti3]]
                elif line[0] == 'vt':
                    u = float(line[1])
                    v = 1 - float(line[2])
                    texCoords += [[u, v]]

    @classmethod
    def combining(cls, meshes: List['Mesh']):
        """ Returns a single mesh object comprising all the individual given meshes.

        :param meshes: the meshes to combine.
        :return: the combined meshes as a single mesh.
        """
        vertices, tris = None, None
        for mesh in meshes:
            if vertices is None:
                vertices = mesh.vertices
                tris = mesh.tris
            else:
                indexOffset = vertices.shape[0]
                vertices = np.append(vertices, mesh.vertices, axis=0)
                tris = np.append(tris, mesh.tris + indexOffset, axis=0)
        return Mesh(vertices, tris)

    @property
    def nVerts(self):
        """The total number of vertices (including nan vertices) in the mesh"""
        return self._vertices.shape[0]

    @property
    def nTris(self):
        """The total number of triangles (including those comprising nan vertices) in the mesh"""
        return self._tris.shape[0]

    @property
    def vertices(self) -> np.ndarray:
        """ The mesh's vertices. For a mesh of n vertices, this is a numpy array of shape (n, 3) containing the x, y, z
        coordinate of each vertex (in the world coordinate frame)."""
        return self._vertices

    @property
    def verticesVec3(self) -> Vec3:
        """ The mesh's vertices (in the world coordinate frame) as a Vec3 object."""
        x = self._vertices[:, 0]
        y = self._vertices[:, 1]
        z = self._vertices[:, 2]
        return Vec3((x, y, z))

    @property
    def tris(self) -> np.ndarray:
        """ The mesh's triangles. Each triangle is defined by the indices of its three vertices in the vertices array.
        For a mesh of m triangles, tris is a numpy array of shape (m, 3)."""
        return self._tris

    @property
    def edges(self) -> np.ndarray:
        """The mesh's edges. Each edge is defined by the indices of the two vertices it connects in the vertices array.
        For a mesh of q edges, this is a numpy array of shape (q, 2)."""
        tris = self._tris
        nTris = tris.shape[0]
        edges = np.empty((3 * nTris, 2), dtype=int)
        edges[:nTris, 0] = tris[:, 0]
        edges[:nTris, 1] = tris[:, 1]
        edges[nTris:2 * nTris, 0] = tris[:, 0]
        edges[nTris:2 * nTris, 1] = tris[:, 2]
        edges[2 * nTris:3 * nTris, 0] = tris[:, 1]
        edges[2 * nTris:3 * nTris, 1] = tris[:, 2]
        edges = np.sort(edges)
        edgesUnique = np.unique(edges, axis=0)
        return edgesUnique

    @property
    def meshO3D(self) -> open3d.t.geometry.TriangleMesh:
        """This mesh as an open3d.t.geometry.TriangleMesh object, in the world coordinate frame."""
        if self._meshO3D is None:
            vertsTranslated = (self._vertices + self._renderingShift.tuple).astype('float32')
            self._meshO3D = open3d.t.geometry.TriangleMesh(vertsTranslated, self._tris)
        return self._meshO3D

    @property
    def meshStrippedOfNans(self) -> 'Mesh':
        """ Returns a new mesh which represents the same surface as this mesh, but all vertices with nan values have
        been removed. This makes no difference to the shape or appearance of a mesh, but reduces data size and can
        mitigate compatibility issues when viewing this mesh with other software. NOTE however that stripping a
        grid-like mesh of nans will remove its grid-like connectivity, and the mesh can no longer be treated as
        grid-like (will cause texture mapping issues if the mesh is linked to a grid-like texture)

        :return: The nan-free mesh
        """
        nanVerts = np.any(np.isnan(self.vertices), axis=1)
        if not np.any(nanVerts):
            return self
        self._gridShape = None
        vertIndexAdjustments = np.cumsum(nanVerts)
        strippedTris = self._tris - vertIndexAdjustments[self._tris]
        strippedVerts = self._vertices[~nanVerts]
        return Mesh(strippedVerts, strippedTris, shift=self._renderingShift)

    @property
    def isGridMesh(self) -> bool:
        """Whether the vertices making up this mesh have 2D grid-like connectivity"""
        return self._gridShape is not None

    @property
    def gridShape(self):
        """If this mesh is a grid mesh (i.e. its vertices have 2D grid-like connectivity), this returns the number
        of rows and columns in that grid (if mesh is not grid-like, returns None)"""
        return self._gridShape

    @property
    def gridOfVertices(self):
        """If this mesh is a grid mesh (i.e. its vertices have 2D grid-like connectivity), this returns its vertices
        as a nxmx3 numpy array (where the mesh grid has n rows and m columns)."""
        if self.isGridMesh:
            nR, nC = self.gridShape
            col, row = np.arange(nC), np.arange(nR)
            col, row = np.meshgrid(col, row)
            flatID = self.vertexGridIndexToFlatID(col, row)
            return self.coordOfVertex(flatID)
        return None

    @property
    def meanVert(self) -> Vec3:
        """The mean of all this mesh's vertices' coordinates"""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        z = self.vertices[:, 2]
        xmean = np.nanmean(x)
        ymean = np.nanmean(y)
        zmean = np.nanmean(z)
        return Vec3((xmean, ymean, zmean))

    @property
    def renderingShift(self):
        """The vector by which this mesh is shifted during ray intersections.

        The purpose of this shift is to improve accuracy by reducing floating point errors when a mesh is a long
        distance from the origin (e.g. as with a small portion of a planetary surface)."""
        return self._renderingShift

    @renderingShift.setter
    def renderingShift(self, shift: Optional[Vec3]):
        """ Updates the vector by which the mesh is shifted during ray intersections.

        The purpose of this shift is to improve accuracy by reducing floating point errors when a mesh is a long
        distance from the origin (e.g. as with a small portion of a planetary surface).
        """
        self._meshO3D = None
        self._sceneO3D = None
        if shift is None:
            self._renderingShift = Vec3.zero()
        else:
            self._renderingShift = shift

    @property
    def textureCoordArray(self) -> Optional[np.ndarray]:
        """tx2 array of texture u,v coordinates for mapping to this mesh's associated texture(s), where t is the number
        of texture coordinates (or None if no texture coordinates have been assigned to this mesh)"""
        return self._textureCoordArray

    @textureCoordArray.setter
    def textureCoordArray(self, array: np.ndarray):
        """ Updates this mesh's texture coordinate array, which is a tx2 array of texture u,v coordinates for mapping to
        this mesh's associated texture(s), where t is the number of texture coordinates

        :param array: tx2 numpy array of u, v coordinates
        """
        self._textureCoordArray = array

    @property
    def triTexIndices(self) -> Optional[np.ndarray]:
        """ mx3 array containing, for each of the mesh's
        triangles, the indices of the texture coordinates in the textureCoordArray associated with each of the
        triangle's three vertices (or None if no vertex texture coord indices have been assigned to this mesh)"""
        return self._triTexIndices

    @triTexIndices.setter
    def triTexIndices(self, array: np.ndarray):
        """ Updates this mesh's triangle texture indices, which is a mx3 array containing, for each of the mesh's
        triangles, the indices of the texture coordinates in the textureCoordArray associated with each of the
        triangle's three vertices.

        :param array: mx3 array of indices
        """
        self._triTexIndices = array

    @property
    def hasVertexTextureMapping(self):
        """Whether this mesh has a texture coordinate mapping defined for its vertices (using wavefront .obj scheme)"""
        if self._textureCoordArray is not None and self._triTexIndices is not None:
            return True
        return False

    def coordOfVertex(self, index: Union[int, np.ndarray]) -> Vec3:
        """ Returns the x,y,z coordinate of the mesh vertex with the given index within the mesh's vertices array

        :param index: index of the vertex
        :return: x, y, z coordinate of the vertex as Vec3 object
        """
        if type(index) is np.ndarray:
            return Vec3.fromNumpyArray(self._vertices[index])
        return Vec3(self._vertices[index])

    def intersect(self, ray: 'Ray', max_dist: float = 1e15) -> Dict[str, np.ndarray]:
        """ Performs an intersection test of the given ray(s) with this mesh and returns the result.

        The returned result is a dictionary containing the hit depths ("t_hit"), hit triangle IDs ("primitive_ids"),
        hit triangle surface normals ("primitive_normals"), hit triangle uv coordinates ("primitive_uvs") and
        geometry id ("geometry_ids").

        The ray mesh intersection is performed using open3D.

        :param ray: The ray or rays (represent multiple rays by using numpy arrays for the ray object's vector elements)
        :param max_dist: Maximum distance of intersection. Any intersections beyond this distance are discarded.
        :return: The ray intersection result.
        """
        ray = Ray(ray.origin + self._renderingShift, ray.d)  # translate ray(s) to mesh's shifted frame
        # (the shifted mesh is used for ray tracing as it improves accuracy by reducing floating point errors)

        if self._sceneO3D is None:
            self._sceneO3D = open3d.t.geometry.RaycastingScene()
            self._sceneO3D.add_triangles(self.meshO3D)

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
                try:
                    rayO3D[:, i + 3] = ray.d[i][notNan]
                except IndexError:
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
            return result

    def coordFromTriUV(self, prim_id: _inp, prim_u: _fnp, prim_v: _fnp) -> 'Vec3':
        """ For a given triangle (a.k.a. primitive) in this mesh, and a given UV coordinate in that triangle's uv
        coordinate frame, this function returns the corresponding xyz coordinate.

        :param prim_id: triangle (primitive) index
        :param prim_u: u coordinate in the primitive's uv coordinate frame
        :param prim_v: v coordinate in the primitive's uv coordinate frame
        :return: corresponding xyz coordinate
        """
        if type(prim_id) is not np.ndarray:
            prim_id = np.array((prim_id,))
        notNan = ~np.isnan(prim_id)
        prim_id_int = np.array(prim_id, dtype=float)
        prim_id_int[~notNan] = 0
        prim_id_int = prim_id_int.astype(int)
        baseSlice = (slice(None),) * prim_id.ndim

        vert1 = self._vertices[self._tris[prim_id_int][baseSlice + (0,)]]
        vert2 = self._vertices[self._tris[prim_id_int][baseSlice + (1,)]]
        vert3 = self._vertices[self._tris[prim_id_int][baseSlice + (2,)]]
        vert1 = Vec3((vert1[baseSlice + (0,)], vert1[baseSlice + (1,)], vert1[baseSlice + (2,)]))
        vert2 = Vec3((vert2[baseSlice + (0,)], vert2[baseSlice + (1,)], vert2[baseSlice + (2,)]))
        vert3 = Vec3((vert3[baseSlice + (0,)], vert3[baseSlice + (1,)], vert3[baseSlice + (2,)]))

        ret = vert1 + (prim_u * (vert2 - vert1)) + (prim_v * (vert3 - vert1))
        ret.x[~notNan] = np.nan
        ret.y[~notNan] = np.nan
        ret.z[~notNan] = np.nan
        return ret

    def saveOBJ(self, outpath: str, sf=1):
        """ Generates an .obj (wavefront) file for this mesh and saves it to the given path

        :param outpath: path to save the .obj file.
        :param sf: scale factor by which to multiply the values of the vertices' coordinates before saving in the .obj
        :param vtc: optional tuple of u and v texture coordinates for each of the vertices in the mesh
        :param mtl: name of mtl file (if any) that describes associated texture
        """
        f = open(outpath, 'w')
        for vertex in self._vertices:
            f.write('v {0} {1} {2}\n'.format(vertex[0] * sf, vertex[1] * sf, vertex[2] * sf))
        if self.hasVertexTextureMapping:
            for coord in self._textureCoordArray:
                u, v = coord
                f.write('vt {0} {1}\n'.format(u, 1 - v))
        for index, tri in enumerate(self._tris):
            val1, val2, val3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
            if self.hasVertexTextureMapping:
                val4 = self._triTexIndices[index][0] + 1
                val5 = self._triTexIndices[index][1] + 1
                val6 = self._triTexIndices[index][2] + 1
                f.write('f {0}/{3} {1}/{4} {2}/{5}\n'.format(val1, val2, val3, val4, val5, val6))
            else:
                f.write('f {0} {1} {2}\n'.format(val1, val2, val3))
        f.close()

    def vertexGridIndexFromFlatID(self, vert_id: _inp) -> Tuple[_inp, _inp]:
        """ If this mesh's vertices have grid-like connectivity, this function returns the grid (column, row) index of a
        vertex given its index within this mesh's 1D vertices array.

        :param vert_id: The index of the vertex within this mesh's vertices array.
        :return: The column and row of the vertex within the point grid.
        """
        if self.isGridMesh:
            shape = self._gridShape
            col = vert_id % shape[1]
            row = np.floor(vert_id / shape[1])
            return col, row
        raise ValueError('Cannot call vertexGridIndexFromFlatID for a trimesh that is not grid-like.')

    def vertexGridIndexToFlatID(self, column: _inp, row: _inp) -> _inp:
        """ If this mesh's vertices have grid-like connectivity, this function returns the index of a vertex within this
        mesh's 1D vertices array given its column and row within the mesh's grid of vertices.

        :param column: The column of the vertex.
        :param row: The row of the vertex.
        :return: The index of the vertex within this mesh's vertices array.
        """
        if self.isGridMesh:
            return row * self._gridShape[1] + column
        raise ValueError('Cannot call vertexGridIndexToFlatID for a trimesh that is not grid-like.')

    def vertexGridUVFromFlatID(self, vert_id: _inp) -> Tuple[_fnp, _fnp]:
        """ If this mesh's vertices have grid-like connectivity, this function returns the grid UV coordinate of
        a vertex given its index within this mesh's 1D vertices array

        :param vert_id: The index of the vertex within this mesh's vertices array.
        :return: The grid UV coordinate
        """
        col, row = self.vertexGridIndexFromFlatID(vert_id)
        nRows, nCols = self._gridShape
        return col / nCols, row / nRows

    def vertexGridUVToFlatID(self, u: _fnp, v: _fnp) -> _inp:
        """ If this mesh's vertices have grid-like connectivity, this function returns the index of a vertex within this
        mesh's 1D vertices array given its uv coordinate within the grid.

        :param u: vertex's grid U coordinate
        :param v: vertex's grid V coordinate
        :return: index of the vertex in the mesh's vertices array
        """
        nRows, nCols = self._gridShape
        col = u * nCols
        if type(col) is np.ndarray:
            col = col.astype(int)
        else:
            col = int(col)
        row = v * nRows
        if type(row) is np.ndarray:
            row = row.astype(int)
        else:
            row = int(row)
        return self.vertexGridIndexToFlatID(col, row)

    def triUVToGridUV(self, prim_id: _inp, prim_u: _fnp, prim_v: _fnp) -> Tuple[_fnp, _fnp]:
        """ If this mesh's vertices have grid-like connectivity:
        Given the id (index in self.tris array) of a primitive (triangle) of this mesh, and a point expressed in
        that triangle's (u, v) coordinate frame, this function returns the point's (U, V) coordinate in the
        frame of the full mesh grid.

        :param prim_id: the primitive's id (i.e. its index in this mesh's tris array)
        :param prim_u: the u coordinate of the point in the primitive's uv frame
        :param prim_v: the v coordinate of the point in the primitive's uv frame
        :return: (U, V) tuple of the point's coordinate in the mesh's uv frame
        """
        if type(prim_id) is not np.ndarray:
            prim_id = np.array((prim_id,))
        notNan = ~np.isnan(prim_id)
        prim_id_int = np.array(prim_id, dtype=float)
        prim_id_int[~notNan] = 0
        prim_id_int = prim_id_int.astype(int)

        vert1 = self.vertexGridIndexFromFlatID(self._tris[prim_id_int][(slice(None),) * prim_id.ndim + (0,)])
        vert2 = self.vertexGridIndexFromFlatID(self._tris[prim_id_int][(slice(None),) * prim_id.ndim + (1,)])
        vert3 = self.vertexGridIndexFromFlatID(self._tris[prim_id_int][(slice(None),) * prim_id.ndim + (2,)])
        vert1 = Vec2(vert1)
        vert2 = Vec2(vert2)
        vert3 = Vec2(vert3)

        u, v = (vert1 + (prim_u * (vert2 - vert1)) + (prim_v * (vert3 - vert1))).tuple
        u[~notNan] = np.nan
        v[~notNan] = np.nan
        return u / self._gridShape[1], v / self._gridShape[0]

    def similarityTransformation(self, verts: List[int], coords: List[Vec3]) -> Tuple[Mat3, Vec3, float]:
        """ This function takes the indices of >=3 of this mesh's vertices along with the 3D coordinates of these
        vertices in an arbitrary coordinate frame and returns the similarity transform (rotation, translation and
        scaling) that transforms points from this mesh's coordinate frame to the arbitrary coordinate frame.

        The similarity transform is returned as a tuple containing a rotation matrix R, translation vector t and
        scale factor s. Points p0 in the mesh's frame are transformed to the new frame by pn=s*R*p0+t.

        :param verts: 1D list of >= 3 indices of vertices in this mesh.
        :param coords: 1D list of coordinates for each vertex in verts in another coordinate frame.
        :return: tuple containing rotation matrix, translation vector and scale factor for transforming points
                from mesh's coordinate frame to the other coordinate frame.
        """
        if len(verts) < 3 or len(verts) != len(coords):
            message = ("verts and coords must be tuples of matching length with >=3 elements, but have lengths {0} and"
                       " {1} respectively".format(len(verts), len(coords)))
            raise ValueError(message)

        nativeCoords = []
        for vert in verts:
            nativeCoords += [self.coordOfVertex(vert)]

        return Frame.similarityTransformFromPoints(nativeCoords, coords)

    def transformed(self, rotation: Mat3, translation: Vec3, scale: float) -> 'Mesh':
        """ Transforms this mesh using the given rotation matrix, translation vector and scale factor (i.e. similarity
        transform) and returns the result.

        :param rotation: rotation matrix.
        :param translation: translation matrix.
        :param scale: scale factor.
        :return: the transformed mesh.
        """
        newVertices = scale * rotation * self.verticesVec3 + translation
        return Mesh.fromVec3(newVertices, self._tris)
