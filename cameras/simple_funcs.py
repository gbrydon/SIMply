# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
"""Module containing simple functions for performing basic calculations and modelling of camera performance"""

import numpy as np
from coremaths import math2
from radiometry import radiometry as rd


def diffractionLimit(d, w):
    """The diffraction limit of an optical camera of given aperture diameter when observing the given wavelength
    (calculated according to the Rayleigh criterion).

    :param d: aperture (entrance pupil) diameter of the camera [m]
    :param w: wavelength of observation [nm]
    :return: diffraction limit [radians]
    """
    return np.arcsin(1.22 * w * 1e-9 / d)


def diffractionLimitingAperture(a, w):
    """Calculates the entrance pupil diameter associated with the given diffraction-limited angular resolution and
    observation wavelength.

    :param a: diffraction-limited angular resolution [radians]
    :param w: observation wavelength [nm]
    :return: entrance pupil diameter [m]
    """
    return 1.22 * w * 1e-9 / np.sin(a)


def pixelSignalToSNR(signal, bg, dark, read):
    """ Gives the signal-to-noise ratio (SNR) in a pixel's measurement

    :param signal: the electron count due to true signal [e-]
    :param bg: the electron count due to background noise [e-]
    :param dark: the electron count due to dark current [e-]
    :param read: the rms read noise [e-]
    :return: the SNR of the pixel's measurement
    """
    return signal / (signal + bg + dark + read ** 2) ** 0.5


def pixelSignalFromSNR(snr, bg, dark, read):
    """ The number of electrons due to true signal counted by a pixel to achieve a given signal-to-noise ratio (SNR)

    :param snr: the desired SNR
    :param bg: the electron count due to background noise [e-]
    :param dark: the electron count due to dark current [e-]
    :param read: the rms read noise [e-]
    :return: the electron count due to true signal [e-]
    """
    _a = 1
    _b = -(snr ** 2)
    _c = -((snr ** 2) * ((bg + dark) + (read ** 2)))
    return math2.quadSolve(_a, _b, _c)[0]


def fluxToElectronCount(flux, t, epd, tr, qe, w):
    """ Calculates the number of signal electrons counted due to a given at-aperture flux

    :param flux: the observed flux [W m^-2]
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :return: number of signal electrons measured due to flux
    """
    a = 0.25 * 3.14159 * epd ** 2  # entrance pupil area [m^2]
    energy = flux * a * tr * t  # total energy incident on pixel due to flux [J]
    n_photons = energy / rd.photonEnergy(w)  # total number of photons incident on pixel due to flux
    return n_photons * qe


def fluxFromElectronCount(ne, t, epd, tr, qe, w):
    """ Calculates the at-aperture flux required to result in a given number of electrons in a pixel

    :param ne: number of electrons
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :return: the observed flux [W m^-2]
    """
    a = 0.25 * 3.14159 * epd ** 2  # entrance pupil area [m^2]
    n_photons = ne / qe  # number of photons incident on pixel due to flux
    energy = n_photons * rd.photonEnergy(w)  # total energy incident on pixel due to flux [J]
    return energy / a / tr / t


def radianceToElectronCount(radiance, t, epd, ifov, tr, qe, w):
    """ Calculates the number of signal electrons counted due to a given observed radiance (assumed uniform over pixel)

    :param radiance: the observed radiance (assumed uniform over pixel) [W m^-2 sr^-1]
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param ifov: the pixel's IFOV [rad]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :return: number of signal electrons measured due to radiance
    """
    flux = radiance * ifov ** 2
    return fluxToElectronCount(flux, t, epd, tr, qe, w)


def radianceFromElectronCount(ne, t, epd, ifov, tr, qe, w):
    """ Calculates the observed radiance required to result in a given number of electrons in a pixel

    :param ne: number of electrons
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param ifov: pixel IFOV [rad]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :return: the observed radiance [W m^-2 sr^-1]
    """
    a = 0.25 * 3.14159 * epd ** 2  # entrance pupil area [m^2]
    omega = ifov ** 2  # solid angle viewed by pixel [str]
    n_photons = ne / qe  # number of photons incident on pixel due to flux
    energy = n_photons * rd.photonEnergy(w)  # total energy incident on pixel due to flux [J]
    return energy / a / omega / tr / t


def fluxToSNR(flux, t, epd, ifov, tr, qe, w, bgrad, jd, nr, dwell=None):
    """ Calculates the SNR of a pixel's measurement as a function of observed flux and noise sources

    :param flux: The flux arriving at the instrument's aperture due to signal [W m^-2]
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param ifov: pixel IFOV [rad]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :param bgrad: total radiance of background sources [W m^-2 sr^-1]
    :param jd: detector dark current [e- s^-1]
    :param nr: read noise RMS [e-]
    :param dwell: dwell time of flux source [s]. If None is passed, tDwell is set equal to t
    :return: SNR of measurement
    """
    if dwell is None:
        dwell = t
    sigcount = fluxToElectronCount(flux, dwell, epd, tr, qe, w)
    bgcount = radianceToElectronCount(bgrad, t, epd, ifov, tr, qe, w)
    return pixelSignalToSNR(sigcount, bgcount, jd * t, nr)


def fluxFromSNR(snr, t, epd, ifov, tr, qe, w, bgrad, jd, nr, dwell=None):
    """ Calculates the at-aperture flux required to achieve a given SNR in a pixel's measurement

    :param snr: the pixel's SNR
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param ifov: pixel IFOV [rad]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :param bgrad: total radiance of background sources [W m^-2 sr^-1]
    :param jd: detector dark current [e- s^-1]
    :param nr: read noise RMS [e-]
    :param dwell: dwell time of flux source [s]. If None is passed, tDwell is set equal to t
    :return: the at-aperture flux observed by the pixel [W m^-2]
    """
    if dwell is None:
        dwell = t
    bgcount = radianceToElectronCount(bgrad, t, epd, ifov, tr, qe, w)
    sigcount = pixelSignalFromSNR(snr, bgcount, jd * t, nr)
    return fluxFromElectronCount(sigcount, dwell, epd, tr, qe, w)


def radianceToSNR(radiance, t, epd, ifov, tr, qe, w, bgrad, jd, nr, dwell=None):
    """ Calculates the SNR of a pixel's measurement as a function of observed flux and noise sources

    :param radiance: The radiance observed by the pixel (assumed uniform over whole pixel) [W m^-2 sr^-1]
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param ifov: pixel IFOV [rad]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :param bgrad: total radiance of background sources [W m^-2 sr^-1]
    :param jd: detector dark current [e- s^-1]
    :param nr: read noise RMS [e-]
    :param dwell: dwell time of radiance [s]. If None is passed, tDwell is set equal to t
    :return: SNR of measurement
    """
    if dwell is None:
        dwell = t
    sigcount = radianceToElectronCount(radiance, dwell, epd, ifov, tr, qe, w)
    bgcount = radianceToElectronCount(bgrad, t, epd, ifov, tr, qe, w)
    return pixelSignalToSNR(sigcount, bgcount, jd * t, nr)


def radianceFromSNR(snr, t, epd, ifov, tr, qe, w, bgrad, jd, nr, dwell=None):
    """ Calculates the observed radiance required to achieve a given SNR in a pixel's measurement

    :param snr: the pixel's SNR
    :param t: exposure time [s]
    :param epd: entrance pupil diameter [m]
    :param ifov: pixel IFOV [rad]
    :param tr: system optical transmission
    :param qe: detector quantum efficiency
    :param w: the effective wavelength of the measurement [nm]
    :param bgrad: total radiance of background sources [W m^-2 sr^-1]
    :param jd: detector dark current [e- s^-1]
    :param nr: read noise RMS [e-]
    :param dwell: dwell time of flux source [s]. If None is passed, tDwell is set equal to t
    :return: the observed radiance [W m^-2 sr^-1]
    """
    if dwell is None:
        dwell = t
    bgcount = radianceToElectronCount(bgrad, t, epd, ifov, tr, qe, w)
    sigcount = pixelSignalFromSNR(snr, bgcount, jd * t, nr)
    return radianceFromElectronCount(sigcount, dwell, epd, ifov, tr, qe, w)
