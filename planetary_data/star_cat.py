# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
import math
import numpy as np
import radiometry.radiometry as rd
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy import units
from simply_utils import constants as consts


def getStarsInRegion(centrera: float, centredec: float, regionrad: float, maglim: float, degrees=True):
    """Returns the details of stars in the given circular region of sky (defined by a centre and an angular radius)
    with brightness greater than the given Vmag limit, retrieved from the ascc catalogue.

    :param centrera: the RA of the centre of the search region
    :param centredec: the dec of the centre of the search region
    :param regionrad: the angular radius of the circular search region
    :param maglim: the maximum V magnitude of star to include
    :param degrees: if set to true, the given angles are taken to be degrees, otherwise radians
    :return: a list of the stars found (each star represented by a dictionary containing 'Vmag', 'Vflux', 'Bmag',
        'Bflux', 'RA' and 'dec', where Vmag and Bmag are the spectral flux (W m^-2 nm^-1) of the star at 550 and 440nm
        respectively, and RA and dec are in radians
    """
    if not degrees:
        centrera = np.degrees(centrera)
        centredec = np.degrees(centredec)
        regionrad = np.degrees(regionrad)
    centreCoord = SkyCoord(ra=centrera, dec=centredec, unit=(units.deg, units.deg), frame='icrs')
    Vizier.ROW_LIMIT = 100000
    returned = Vizier.query_region(centreCoord, width='{}d'.format(2 * regionrad), catalog='ascc')
    if not returned:
        return []
    candidates = returned[0]
    results = candidates[candidates['Vmag'] < maglim]
    starList = []

    def nanIfMasked(val):
        if np.ma.is_masked(val):
            return np.nan
        return val
    for result in results:
        ra = nanIfMasked(math.radians(result['RAJ2000']))
        dec = nanIfMasked(math.radians(result['DEJ2000']))
        vMag = nanIfMasked(result['Vmag'])
        vFlux = nanIfMasked(rd.fluxFromMag(vMag, consts.vmagZeroFlux))
        bMag = nanIfMasked(result['Bmag'])
        bFlux = nanIfMasked(rd.fluxFromMag(bMag, consts.bmagZeroFlux))
        starList += [{'Vmag': vMag, 'Vflux': vFlux, 'Bmag': bMag, 'Bflux': bFlux, 'RA': ra, 'dec': dec}]
    return starList
