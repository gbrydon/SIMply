# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module containing functions for rendering simulated images."""

import numpy as np
from coremaths import math2
from coremaths.vector import Vec3
from coremaths.ray import Ray
from rendering.renderables import RenderableObject, RenderableScene
from cameras.cameras import Camera, CameraTwoWay
from radiometry import radiometry as rd
from radiometry.radiometry import SpectralDensityCurve as SDC
from typing import List, Optional, Tuple, Union


class Renderer:
    """Class containing functions for rendering images of renderable objects/scenes."""

    # Type aliases:
    _Renderable = Union[RenderableObject, RenderableScene]
    _RS = RenderableScene
    _Cam = CameraTwoWay
    _Cams = Union[_Cam, List[_Cam]]
    _W2 = Tuple[float, float]
    _W3 = Tuple[float, float, float]
    _radWs = List[Union[_W2, _W3]]
    _imWs = List[_W3]
    _t = Union[float, List[float]]
    _Lims = Tuple[int, int, int, int]
    _nanv = Union[float, Tuple[float, float, float]]

    @staticmethod
    def getView(scene: _Renderable, ray: Ray, n_shad=1, e=5e-4):
        """ Returns the view that the given rays have of the given scene/object. This function returns a tuple
        containing (1) the intersection dictionary returned by performing an intersection of the rays with the scene
        (see RenderableScene.intersect(...) documentation for further details) and (2) a view dictionary containing, for
        each ray, the xyz coordinate of intersection (key='p_hit'), normalised vector from intersection point to light
        source (key='to_light'), whether the point of intersection is in shadow from the scene's light source for each
        secondary ray (key='shadow'). More details on these contents of the view dictionary are given below.

        View dictionary details:
        For a ray object with numpy shape s (e.g. s would be (1000, 1000) for rendering an image with 1000x1000 primary
        rays), view['p_hit'] is a Vec3 object with numpy shape s; view['to_light] is a Vec3 object with numpy shape
        (n_shad,) + s if n_shad > 1, and shape s if n_shad <= 1; view['shadow'] is a numpy array with shape
        (n_shad,) + s if n_shad > 1, and shape s if n_shad <= 1.

        :param scene: the scene
        :param ray: the ray(s) (for multiple rays use a single Ray object with numpy arrays as its vector components)
        :param n_shad: number of secondary rays to trace per primary ray for calculating shadowing
        :param e: epsilon value for minimising self-shadowing artefacts
        :return: tuple containing intersection dictionary and view dictionary
        """
        intersection = scene.intersect(ray)
        nRaw = intersection['primitive_normals']
        n = Vec3.fromNumpyArray(nRaw)

        pHit = scene.pIntersection(intersection)
        view = {'p_hit': pHit}

        if scene.light is not None:  # scene has a light, so get details of illumination at the viewed scene points
            pHit = pHit + e * -ray.d.projectedOnto(n).norm  # shift pHit to mitigate floating point error self-shadowing
            if n_shad == 0:  # no shadow testing
                shadow = np.full(ray.numpyShape, 0)
                ls = (scene.light.pos - pHit).norm
            else:
                if n_shad == 1:  # shadow testing with 1 secondary ray per primary ray
                    rayToLight = scene.light.traceRayToCentre(pHit)
                else:  # shadow testing with multiple secondary rays per primary ray
                    x = np.repeat(pHit.x[None], n_shad, axis=0)
                    y = np.repeat(pHit.y[None], n_shad, axis=0)
                    z = np.repeat(pHit.z[None], n_shad, axis=0)
                    origShape = pHit.numpyShape
                    stackShape = (n_shad,) + origShape
                    pAdjusted = Vec3((x, y, z))
                    index = np.arange(n_shad).repeat(origShape[0] * origShape[1]).reshape(stackShape)
                    rayToLight = scene.light.traceRayDistributed(pAdjusted, index, n_shad)
                ls = rayToLight.d
                retShadTest = scene.intersect(rayToLight)
                shadow = ~np.isnan(retShadTest['t_hit'])
            view['to_light'] = ls
            view['shadow'] = shadow

        return intersection, view

    @staticmethod
    def depth(scene: _Renderable, camera: _Cam, lim: _Lims = None):
        """ Renders a depth image of the given scene or model as viewed by the given camera.

        A depth image gives the distance [m] from the camera's entrance pupil to the viewed point of the scene for each
        primary ray.

        :param scene: model to render
        :param camera: camera whose view will be rendered
        :param lim: provide a (minCol, maxCol, minRow, maxRow) tuple to limit the ray tracing calculations to that area
            of the detector. All other regions of the detector will be treated as seeing zero radiance. Use this to
            increase render speed and/or reduce memory required for rendering the radiance image when for example all
            sources of radiance are known to sit within a limited region of the image.
        :return: depth image [m]
        """
        rays = camera.pixelsLOS(region=lim)
        ret = scene.intersect(rays)
        d = ret['t_hit']

        if lim is not None:
            fullSize = np.full((camera.dr, camera.dc), np.nan)
            minC, maxC, minR, maxR = lim
            fullSize[minR:maxR, minC:maxC] = d
            d = fullSize

        return d

    @staticmethod
    def texture(scene: _Renderable, camera: _Cam, sf=1, lim: _Lims = None, nanv: _nanv = None, chan='g') -> np.ndarray:
        """ Renders an image of the given scene/model as viewed by the given camera, in which the scene's appearance is
        mapped directly from its associated textures (rays observing objects with no texture are assigned a value of 0).

        :param scene: renderable scene/model
        :param camera: camera
        :param sf: sampling factor controlling number of rays to trace per camera pixel (no. rays per pixel = sf^2), for
            improving rendering quality with subpixel sampling.
        :param lim: provide a (minCol, maxCol, minRow, maxRow) tuple to limit the ray tracing calculations to that area
            of the detector. All other regions of the detector will be treated as seeing zero radiance. Use this to
            increase render speed and/or reduce memory required for rendering the radiance image when for example all
            sources of radiance are known to sit within a limited region of the image.
        :param nanv: value to use in place of nan (i.e. areas in image where no object is observed) in the final image.
            Leave as None to keep nan values. If chan='rgb', nanv can be an (r, g, b) tuple or a single value.
        :param chan: string indicating how to handle RGB textures. Passing 'rgb' results in an rgb (3-channel) image
            where objects with rgb textures will be in colour, and objects with single-channel textures will be grey.
            Passing 'r', 'g' or 'b' results in a single-channel greyscale image where either the red, green or blue
            channels of any rgb textures in the scene are used.
        :return: texture image of the scene
        """

        rays = camera.pixelsLOS(sf=sf, region=lim)
        ret = scene.intersect(rays)

        def getTexImageForChannel(channel: int):
            texValue = scene.textureValue(ret, rgb_channel=channel)
            if sf != 1:
                texValue = math2.binNumpyArray2D(texValue, sf) / sf ** 2

            if lim is not None:
                fullSize = np.full((camera.dr * sf, camera.dc * sf), np.nan)
                minC, maxC, minR, maxR = lim
                fullSize[minR * sf:maxR * sf, minC * sf:maxC * sf] = texValue
                texValue = fullSize

            if nanv is not None:
                if type(nanv) is tuple:
                    nanValue = nanv[channel - 1]
                else:
                    nanValue = nanv
                texValue[np.isnan(texValue)] = nanValue

            texValue = camera.applyPSF(texValue)

            return texValue

        if chan == 'rgb':
            imgR = getTexImageForChannel(1)
            imgG = getTexImageForChannel(2)
            imgB = getTexImageForChannel(3)
            ret = np.empty(imgR.shape + (3,))
            ret[..., 0] = imgR
            ret[..., 1] = imgG
            ret[..., 2] = imgB
            return ret
        else:
            rgb_channel = 2
            if chan == 'r':
                rgb_channel = 1
            elif chan == 'b':
                rgb_channel = 3
            return getTexImageForChannel(rgb_channel)

    @staticmethod
    def shadow(scene: _RS, camera: _Cam, sf=1, n_shad=1, e=5e-4, lim: _Lims = None):
        """ For the given renderable scene and camera, this function renders an image showing any regions
        of the scene that are in shadow.

        :param scene: the scene
        :param camera: the camera
        :param sf: sampling factor controlling number of primary rays to trace per camera pixel
            (no. rays per pixel = sf^2), for improving rendering quality with subpixel sampling.
        :param n_shad: number of secondary rays to trace per primary ray (for calculating shadows).
        :param e: epsilon value for mitigating rounding-error self-shadowing artefacts.
        :param lim: provide a (minCol, maxCol, minRow, maxRow) tuple to limit the ray tracing calculations to that area
            of the detector. All other regions of the detector will be treated as seeing zero radiance. Use this to
            increase render speed and/or reduce memory required for rendering the radiance image when for example all
            sources of radiance are known to sit within a limited region of the image.
        :return: shadow mask
        """
        if scene.light is None:
            raise ValueError('scene must have a light source to render a shadow image')
        if n_shad < 0:
            raise ValueError('number of shadow test rays n_shad must be an integer >= 0')

        rays = camera.pixelsLOS(sf=sf, region=lim)

        _, view = Renderer.getView(scene, rays, n_shad, e)
        shadow = view['shadow']

        if n_shad > 1:
            shadow = np.sum(shadow, axis=0) / n_shad

        if lim is not None:
            fullSize = np.zeros((camera.dr * sf, camera.dc * sf))
            minC, maxC, minR, maxR = lim
            fullSize[minR * sf:maxR * sf, minC * sf:maxC * sf] = shadow
            shadow = fullSize

        return shadow

    @staticmethod
    def radiance(scene: _RS, camera: _Cam, w: _W2, sf=1, n_shad=1, e=5e-4, lim: _Lims = None, raw=False):
        """ Generates a physically-based at-sensor radiance image of the given scene, as viewed by the given camera.

        A radiance image is the radiance [W m^-2 sr^-1] of the region of the scene being viewed by each pixel
        (or subpixel) of the camera over the given wavelength range, given the illumination of the scene by its light.
        This function returns a single image giving the total radiance over the given wavelength range.
        For multispectral rendering (rendering multiple wavebands of the same scene under the same viewing geometry),
        use Renderer.radianceMS function.

        :param scene: the scene to render
        :param camera: the camera
        :param w: tuple of minimum and maximum wavelengths [nm] of wavelength range over which to calculate the radiance
        :param sf:  sampling factor controlling number of rays to trace per camera pixel (no. rays per pixel = sf^2),
            for improving rendering quality with subpixel sampling.
        :param n_shad: number of secondary rays to trace per primary ray (where a primary ray is a ray originating from
            the camera) for simulating shadows.
        :param e: epsilon value for minimising self-shadowing artefacts
        :param lim: provide a (minCol, maxCol, minRow, maxRow) tuple to limit the ray tracing calculations to that area
            of the detector. All other regions of the detector will be treated as seeing zero radiance. Use this to
            increase render speed and/or reduce memory required for rendering the radiance image when for example all
            sources of radiance are known to sit within a limited region of the image.
        :param raw: whether the returned array should contain the raw radiance values for each traced ray, or a single
                radiance value for each camera pixel by combining sub-pixel rays and applying PSF.
        :return: the physically rendered radiance image [W m^-2 sr^-1]
        """
        if not scene.physicallyRenderable:
            raise ValueError('scene must have a light source, and BRDFs assigned to objects, to be physically rendered')

        return Renderer.radianceMS(scene, camera, [w], sf, n_shad, e, lim, raw)[0]

    @staticmethod
    def radianceMS(scene: _RS, camera: _Cam, w: _radWs, sf=1, n_shad=1, e=5e-4, lim: _Lims = None, raw=False):
        """ Generates a multispectral set of physically-based at-sensor radiance images of the given scene,
        as viewed by the given camera using the given wavebands.

        A radiance image is the radiance [W m^-2 sr^-1] of the region of the scene being viewed by each pixel (or
        subpixel) of the camera over the image's associated wavelength range, given the illumination of the scene by
        its light. This function returns a list of these images, one for each of the given wavebands. If only a single
        waveband is desired (i.e. monochrome image), consider using Renderer.radiance(...) function instead.

        :param scene: the scene to render
        :param camera: the camera
        :param w: list of wavebands to render radiance images for. Each waveband should be either a length-2 tuple
            giving the waveband's maximum and minimum wavelengths, or a length-3 tuple giving the waveband's minimum
            wavelength, its effective wavelength (neccessary if using spectral BRDFs), and its maximum wavelength [nm].
        :param sf: sampling factor controlling number of rays to trace per camera pixel (no. rays per pixel = sf^2),
            for improving rendering quality with subpixel sampling.
        :param n_shad: number of secondary rays to trace per primary ray (where a primary ray is a ray originating from
            the camera) for simulating shadows.
        :param e: epsilon value for minimising self-shadowing artefacts
        :param lim: provide a (minCol, maxCol, minRow, maxRow) tuple to limit the ray tracing calculations to that area
            of the detector. All other regions of the detector will be treated as seeing zero radiance. Use this to
            increase render speed and/or reduce memory required for rendering the radiance image when for example
            all sources of radiance are known to sit within a limited region of the image.
        :param raw: whether the returned array should contain the raw radiance values for each traced ray, or a single
            radiance value for each camera pixel by combining sub-pixel rays and applying PSF.
        :return: list containing a physically rendered radiance image [W m^-2 sr^-1] for each given waveband.
        """
        if not scene.physicallyRenderable:
            raise ValueError('scene must have a light source, and BRDFs assigned to objects, to be physically rendered')

        rays = camera.pixelsLOS(sf=sf, region=lim)

        ret: List[np.ndarray] = []

        intersection, view = Renderer.getView(scene, rays, n_shad, e)

        nRaw = intersection['primitive_normals']
        n = Vec3.fromNumpyArray(nRaw)
        ls = view['to_light']

        for waveband in w:
            if len(waveband) == 3:
                brdf = scene.brdfEvaluated(intersection, n, ls, -rays.d, waveband[1])
            else:
                brdf = scene.brdfEvaluated(intersection, n, ls, -rays.d)

            f = scene.light.fluxDensity(view['p_hit'], waveband[0], waveband[-1])
            r = rd.surfaceRadiance(f, n, ls, -rays.d, brdf)
            r[view['shadow']] = 0

            if n_shad > 1:
                r = np.sum(r, axis=0) / n_shad

            r[np.isnan(r)] = 0

            if lim is not None:
                fullSize = np.zeros((camera.dr * sf, camera.dc * sf))
                minC, maxC, minR, maxR = lim
                fullSize[minR * sf:maxR * sf, minC * sf:maxC * sf] = r
                r = fullSize

            if raw:
                ret += [r]
            else:
                r = camera.applyPSF(r)
                r = camera.downsampleToDetectorPixels(r) / sf ** 2
                ret += [r]

        return ret

    @staticmethod
    def image(scene: _RS, camera: _Cam, t: float, w: _W3, sf=1, n_shad=1, e=5e-4, lim: _Lims = None, roi: _Lims = None):
        """ Generates and returns a physically-based digital (ADU) image of the given scene, as viewed by the given
        camera. This function also returns the associated at-aperture radiance [W m^-2 sr^-1] image (as per
        radiance(...) function).

        For multispectral images see Renderer.imageMS.

        :param scene: the scene to render
        :param camera: the camera
        :param t: exposure time [s]
        :param w: length-3 tuple containing minimum, effective and maximum wavelengths of image waveband [nm]
        :param sf: sampling factor controlling number of rays to trace per camera pixel (no. rays per pixel = sf^2),
            for improving rendering quality with subpixel sampling.
        :param n_shad: number of secondary rays to trace per primary ray (where a primary ray is a ray originating from
            the camera) for simulating shadows.
        :param e: epsilon value for minimising self-shadowing artefacts
        :param lim: provide a (minCol, maxCol, minRow, maxRow) tuple to limit the ray tracing calculations to that area
            of the detector. All other regions of the detector will be treated as seeing zero radiance. Use this to
            increase render speed and/or reduce memory required for rendering the radiance image when for example
            all sources of radiance are known to sit within a limited region of the image.
        :param roi: provide a (minCol, maxCol, minRow, maxRow) region of interest to limit the rendered image to that
            region of the detector
        :return: tuple containing the digital image and at-aperture radiance image.
        """
        digImage, radImage = Renderer.imageMS(scene, camera, t, [w], sf, n_shad, e, lim, roi)
        return digImage[0], radImage[0]

    @staticmethod
    def imageMS(scene: _RS, camera: _Cams, t: _t, w: _imWs, sf=1, n_shad=1, e=5e-4, lim: _Lims = None, roi: _Lims = None):
        """ Generates and returns a multispectral set of physically-based digital (ADU) images of the given scene
        (one image per given waveband), as viewed by the given camera. This function also returns the at-aperture
        radiance [W m^-2 sr^-1] image associated with each digital image (as per the radianceMS(...) function).

        This function renders one image for each waveband in the list passed for argument w.

        If a different camera response needs to be simulated for each of these wavelengths (e.g. different quantum
        efficiency), provide a list of cameras with the same length as w for the camera argument, and set the internal
        parameters (e.g. qe, nr, psfSigma) to the appropriate values in each for their corresponding waveband.
        If multiple cameras are provided, they will all be treated as viewing the scene with an identical geometry,
        (given by the first camera in the list). If each spectral channel needs to have a different viewing geometry
        (e.g. different camera position or FOV), separate calls to Renderer.image or Renderer.imageMS are required for
        each viewing geometry.

        If the given scene contains any objects with spectral BRDFs, these spectral BRDFs must have wavelengths matching
        the effective wavelengths of the given wavebands.

        If a different exposure time is required for each spectral channel, provide a list of exposure times for
        argument t (this list should be the same length as the waveband list w), otherwise provide a single float value
        to be used for all wavebands.

        :param scene: the scene to render.
        :param camera: either a single camera (if camera response is the same for all spectral channels) or a list of
            cameras (1 per spectral channel) if camera response is different for each spectral channel.
        :param t: exposure time [s] (either a single value to be used for all spectral channels, or a list of exposure
            times, one for each spectral channel).
        :param w: list of wavebands to render images for. Each waveband is a length-3 tuple giving the
            waveband's minimum wavelength, its effective wavelength, and its maximum wavelength [nm].
        :param sf: sampling factor controlling number of rays to trace per camera pixel (no. rays per pixel = sf^2),
            for improving rendering quality with subpixel sampling.
        :param n_shad: number of secondary rays to trace per primary ray (where a primary ray is a ray originating from
            the camera) for simulating shadows.
        :param e: epsilon value for minimising self-shadowing artefacts
        :param lim: provide a (minCol, maxCol, minRow, maxRow) tuple to limit the ray tracing calculations to that area
            of the detector. All other regions of the detector will be treated as seeing zero radiance. Use this to
            increase render speed and/or reduce memory required for rendering the radiance image when for example
            all sources of radiance are known to sit within a limited region of the image.
        :param roi: provide a (minCol, maxCol, minRow, maxRow) region of interest to limit the rendered image to that
            region of the detector.
        :return: tuple containing: (1) list containing the rendered digital images for each waveband; (2) list
            containing the rendered at-aperture radiance images for each waveband.
        """
        if type(camera) is list:
            if len(camera) != len(w):
                message = ("must provide either a single camera (to be used for all wavebands), or a list containing a "
                           "camera per waveband, but {0} cameras and {1} wavebands "
                           "were provided.").format(len(camera), len(w))
                raise ValueError(message)
            refCam = camera[0]
        else:
            refCam = camera
        if type(t) is list and len(t) != len(w):
            message = ("must provide either a single exposure time (to be used for all wavebands), or a list containing"
                       " an exposure time per waveband, but {0} exposure times and {1} wavebands were "
                       "provided").format(len(t), len(w))
            raise ValueError(message)

        digitalImages: list[np.ndarray] = []
        radianceImages: list[np.ndarray] = []

        radiances = Renderer.radianceMS(scene, refCam, w, sf, n_shad, e, lim, raw=True)
        for index, radiance in enumerate(radiances):
            if type(camera) is list:
                cam = camera[index]
            else:
                cam = camera
            if type(t) is list:
                tExp = t[index]
            else:
                tExp = t

            flux = cam.convertRadianceImageToEquivalentFlux(radiance)
            im = cam.image(flux, tExp, w[index][1])

            radianceImage = math2.binNumpyArray2D(radiance, sf)

            if roi is not None:
                digitalImages += [im[roi[2]:roi[3], roi[0]:roi[1]]]
                radianceImages += [radianceImage[roi[2]:roi[3], roi[0]:roi[1]]]
            else:
                digitalImages += [im]
                radianceImages += [radianceImage]

        return digitalImages, radianceImages


class PointSources:
    """ Class containing functions for rendering images of flux point sources."""

    # Type aliases:
    _flx = Union[float, SDC]
    _W2 = Optional[Tuple[float, float]]
    _W3 = Union[float, Tuple[float, float, float]]

    @staticmethod
    def flux(point_sources: List[Tuple[_flx, Vec3]], cam: CameraTwoWay, w: _W2 = None, sf=1, raw=False):
        """ Generates a physically-based at-sensor flux image of the given flux point sources, as viewed by the given
        camera.

        A flux image is the flux [W m^-2] of the region of the scene being viewed by each pixel
        (or subpixel) of the camera over the given wavelength range.

        :param point_sources: list of flux point sources, where each point source is given as a length-2 tuple whose
            first element is either the total integrated flux from the point source over the wavelength range of
            interest, or a spectral density curve of the point source's spectral flux, and the second element is the
            location of the point source in world coordinates.
        :param cam: the camera viewing the scene.
        :param w: if any of the given point sources' fluxes are given as spectral density curves, this argument must be
            a length-2 tuple giving the minimum and maximum wavelengths [nm] between which the flux image is calculated,
             otherwise leave as None.
        :param sf: integer factor by which the flux image's number of pixels is upscaled from the given camera's
            detector.
        :param raw: whether the returned array should contain the raw flux values for eac sampled sub-pixel, or a single
            flux value for each camera pixel by combining sub-pixel fluxes and applying PSF.
        :return: the flux image.
        """
        fluxIm = np.zeros((cam.dr * sf, cam.dc * sf))

        for pointSource in point_sources:
            flux, pos = pointSource
            if type(flux) is SDC:
                flux = flux.integrated(w[0], w[1])
            col, row = cam.worldToImage(pos, cull=True)
            if col is not None and row is not None:
                fluxIm[int(row) * sf, int(col) * sf] += flux

        if raw:
            return fluxIm
        else:
            f = cam.applyPSF(fluxIm)
            f = cam.downsampleToDetectorPixels(f) / sf ** 2
            return f

    @staticmethod
    def radiance(point_sources: List[Tuple[_flx, Vec3]], cam: CameraTwoWay, w: _W2 = None, sf=1, raw=False):
        """ Generates a physically-based image of the effective radiance that the given camera is observing when
        viewing the given flux point sources.

        A radiance image is the radiance [W m^-2 sr^-1] of the region of the scene being viewed by each pixel
        (or subpixel) of the camera over the given wavelength range.

        :param point_sources: list of flux point sources, where each point source is given as a length-2 tuple whose
            first element is either the total integrated flux from the point source over the wavelength range of
            interest, or a spectral density curve of the point source's spectral flux, and the second element is the
            location of the point source in world coordinates.
        :param cam: the camera viewing the scene.
        :param w: if any of the given point sources' fluxes are given as spectral density curves, this argument must be
            a length-2 tuple giving the minimum and maximum wavelengths [nm] between which the flux image is calculated,
             otherwise leave as None.
        :param sf: integer factor by which the flux image's number of pixels is upscaled from the given camera's
            detector.
        :param raw: whether the returned array should contain the raw radiance values for eac sampled sub-pixel, or a
            single radiance value for each camera pixel by combining sub-pixel radiances and applying PSF.
        :return: the radiance image.
        """
        flux = PointSources.flux(point_sources, cam, w, sf, raw)
        return cam.convertFluxImageToEquivalentRadiance(flux)

    @staticmethod
    def image(point_sources: List[Tuple[float, Vec3]], cam: CameraTwoWay, t: float, w: _W3, sf=1):
        """ Generates a physically-based digital (ADU) image of the given scene, as viewed by the given camera.

        :param point_sources: list of flux point sources, where each point source is given as a length-2 tuple whose
            first element is either the total integrated flux from the point source over the wavelength range of
            interest, or a spectral density curve of the point source's spectral flux, and the second element is the
            location of the point source in world coordinates.
        :param cam: the camera viewing the scene.
        :param t: exposure time [s].
        :param w: waveband of the image - this is either (1) a 3-length tuple giving min, effective and max wavelength
            (if any of the given point sources' fluxes are given as spectral density curves) or (2) a float giving the
            effective wavelength at which the imaging is occuring (if all the given point sources' fluxes are given as
            single integrated values).
        :param sf: integer factor by which the flux image's number of pixels is upscaled from the given camera's
            detector during rendering. Larger sf gives more realistic image.
        :return: the image.
        """
        if type(w) is tuple:
            flux = PointSources.flux(point_sources, cam, (w[0], w[2]), sf)
            return cam.image(flux, t, w[1])
        else:
            flux = PointSources.flux(point_sources, cam, None, sf)
            return cam.image(flux, t, w)

