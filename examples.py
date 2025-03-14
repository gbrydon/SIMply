"""Module containing some simple examples demonstrating how to use the software to render images."""

# imports (see examples to see what they're used for)
import numpy as np
from cameras.cameras import Camera
from coremaths.vector import Vec3
from coremaths.frame import Frame
from coremaths.geometry import Rectangle, Spheroid
from rendering.renderables import RenderableObject, RenderableScene
from rendering.lights import Light
from rendering.renderer import Renderer
from rendering.meshes import Mesh
from rendering.textures import Texture
from radiometry.reflectance_funcs import BRDF, TexturedBRDF as tBRDF
from simply_utils import paths, constants as consts
import matplotlib.pyplot as plt


def renderSpheroid(display=True):
    """This example sets up a simple scene containing a spheroid and rectangle illuminated by the sun, and renders an
    image of the scene.

    Steps demonstrated in this example:
        - defining a renderable object from a geometric primitive and BRDF
        - defining a renderable scene with renderable objects and a light
        - defining a pinhole camera; adjusting the camera's pose; adjusting camera parameters
        - rendering and previewing a digital image

    :param display: whether to display the rendered image
    """
    # define spheroid surface
    surface = Spheroid(Frame.world(), 1.2, 1.5, 1)
    # define spheroid's reflection properties using a Phong BRDF
    brdf = BRDF.phong(0.01, 0.005, 1)
    # initialise the renderable object
    spheroid = RenderableObject.renderablePrimitive(surface, brdf)

    # define rectangle surface
    surface = Rectangle(Frame.withW(Vec3((1, 0, 0)), origin=Vec3((-10, 0, 0))), 100, 100)
    # define rectangle's reflection properties using a lambertian BRDF
    brdf = BRDF.lambert(0.01)
    # initialise the renderable object
    rectangle = RenderableObject.renderablePrimitive(surface, brdf)

    # set up scene's light source
    au = 1.5e11
    light = Light.sunPointSource(au * Vec3((2, -1, 0)).norm)

    # initialise scene
    scene = RenderableScene([spheroid, rectangle], light)

    # set up camera
    camera = Camera.pinhole((30, 30), 500, 500)
    # update camera's frame. This positions the camera at [20, 0, 0] and points it along the -x direction,
    # with its horizontal image axis aligned with the y-axis of the world frame
    camera.frame = Frame.withW(Vec3((-1, 0, 0)), origin=Vec3((20, 0, 0)), u=Vec3((0, 1, 0)))
    camera.epd = 2e-3
    camera.psfSigma = 0.6
    camera.nr = 40

    # render the digital image
    img, _ = Renderer.image(scene, camera, 0.2, [500, 502, 504], sf=2, n_shad=2)

    if display:
        plt.imshow(img, cmap='gray')
        plt.show()

    return img


def renderMesh1(display=True):
    """ This example loads a mesh from an .obj file and renders an image of it. A spatially-varying brdf is applied
    to the mesh, as well as a texture image which can be used for texture-based (non-physically-based) rendering,
    using the .obj file's texture mapping information (see renderMesh2 example for alternative texture mapping methods).

    Things demonstrated in this example:
        - loading a mesh from an .obj
        - defining a renderable object from a mesh, a brdf and a texture image
        - defining a renderable scene with a renderable object and a light
        - defining a pinhole camera; adjusting the camera's pose; adjusting camera parameters
        - rendering a digital image, texture image and depth image of a scene

    :param display: whether to display the rendered images
    """

    # load a triangle mesh from an .obj (wavefront) file:
    meshPath = paths.dataFilePath(['input', 'examples', 'model1'], 'surface.obj')
    mesh = Mesh.loadFromOBJ(meshPath)
    # in this example, the wavefront file includes vertex texture mapping information for mapping textures to the mesh's
    # surface (and vice versa). Two texture image files are included for demonstrating this:
    # load the image describing a spatially-varying BRDF parameter and initialise a textured BRDF from it:
    brdfPath = paths.dataFilePath(['input', 'examples', 'model1'], 'lambert_ref.npy')
    brdf_vals = np.load(brdfPath)
    brdf_tex = Texture(brdf_vals)
    brdf = tBRDF(BRDF.lambert, (brdf_tex,))
    # use the image named texture.jpg to define the renderable object's texture image:
    texPath = paths.dataFilePath(['input', 'examples', 'model1'], 'texture.jpg')
    tex = Texture(texPath)
    # initialise the renderable object:
    renderable = RenderableObject.renderableMesh(mesh, brdf, tex)

    # set up the scene's light:
    au = consts.au
    light = Light.sunPointSource(au * Vec3((3, 2, 1)).norm)

    # set up the scene:
    scene = RenderableScene([renderable], light)

    # set up the camera
    cam = Camera.pinhole((40, 40), 500, 500)
    # update the camera's frame to position the camera at [400, 0, 0] with principal axis along the -x direction and
    # horizontal image axis aligned with z-axis of the world frame
    cam.frame = Frame.withW(Vec3((-1, 0, 0)), origin=Vec3((400, 0, 0)), u=Vec3((0, 0, 1)))
    # update the camera's quantum efficiency and optical transmission parameters:
    cam.qe = 0.5
    cam.tr = 0.8

    # render a digital image of the scene captured with exposure time 0.05s, waveband 500-502nm:
    img, _ = Renderer.image(scene, cam, 0.05, [500, 501, 502], sf=1, n_shad=1)
    # render a texture image of the scene:
    tex_img = Renderer.texture(scene, cam)
    # render a depth image of the scene:
    depth = Renderer.depth(scene, cam)

    if display:
        plt.imshow(img, cmap='gray')
        plt.show()
        plt.imshow(tex_img, cmap='gray')
        plt.show()
        plt.imshow(depth)
        plt.show()

    return img


def renderMesh2(display=True):
    """ This example loads a mesh from an .obj file and renders an image of it. A spatially-varying brdf is applied
    to the mesh, as well as a texture image which can be used for texture-based (non-physically-based) rendering,
    using planetocentric texture mapping.

    Things demonstrated in this example:
        - loading a mesh from an .obj
        - defining a renderable object from a mesh, a brdf and a texture image (using planetocentric textures)
        - defining a renderable scene with a renderable object and a light
        - defining a pinhole camera; adjusting the camera's pose; adjusting camera parameters
        - rendering a digital image and texture image

    :param display: whether to display the rendered images
    :return: the rendered digital image
    """

    # load a triangle mesh from an .obj (wavefront) file:
    meshPath = paths.dataFilePath(['input', 'examples', 'model1'], 'surface.obj')
    mesh = Mesh.loadFromOBJ(meshPath)
    # remove the vertex texture mapping information that was loaded from the .obj file (to replicate a mesh for which no
    # vertex texture mapping info is available):
    mesh.textureCoordArray = None
    mesh.triTexIndices = None

    # use planetocentric textures to apply a textured BRDF and texture image to the mesh (see TexturePlanetocentric
    # class for more details):
    brdfPath = paths.dataFilePath(['input', 'examples', 'model1'], 'lambert_ref.npy')
    brdf_vals = np.load(brdfPath)
    brdf_tex = Texture.planetocentric(brdf_vals, (0, 360, -90, 90))
    brdf = tBRDF(BRDF.lambert, (brdf_tex,))
    # use the image named texture.jpg to define the renderable object's texture image:
    texPath = paths.dataFilePath(['input', 'examples', 'model1'], 'texture.jpg')
    tex = Texture.planetocentric(texPath, (0, 360, -90, 90))
    # initialise the renderable object:
    renderable = RenderableObject.renderableMesh(mesh, brdf, tex)

    # set up the scene's light:
    au = consts.au
    light = Light.sunPointSource(au * Vec3((3, 2, 1)).norm)

    # set up the scene:
    scene = RenderableScene([renderable], light)

    # set up the camera
    cam = Camera.pinhole((40, 40), 500, 500)
    # update the camera's frame to position the camera at [0, 400, 0] with principal axis along the -y direction
    cam.frame = Frame.withW(Vec3((0, -1, 0)), origin=Vec3((0, 400, 0)))

    # render a digital image of the scene captured with exposure time 0.05s, waveband 500-502nm:
    img, _ = Renderer.image(scene, cam, 0.05, [500, 501, 502], sf=1, n_shad=1)
    # render a texture image of the scene:
    tex_img = Renderer.texture(scene, cam)

    if display:
        plt.imshow(img, cmap='gray')
        plt.show()
        plt.imshow(tex_img, cmap='gray')
        plt.show()

    return img


# renderSpheroid()
# renderMesh1()
# renderMesh2()
