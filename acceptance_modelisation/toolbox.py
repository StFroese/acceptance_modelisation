import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, SkyOffsetFrame, RepresentationMapping
from astropy.coordinates.earth import OMEGA_EARTH, EarthLocation
from astropy.time import Time

from astropy.coordinates.representation import (
    PhysicsSphericalRepresentation,
    PhysicsSphericalDifferential,
)
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_transpose

from astropy.coordinates.transformations import (
    DynamicMatrixTransform,
    FunctionTransform,
)

from astropy.coordinates.attributes import CoordinateAttribute, QuantityAttribute
from astropy.coordinates.baseframe import frame_transform_graph


def compute_rotation_speed_fov(
    time_evaluation: Time,
    pointing_sky: SkyCoord,
    observatory_earth_location: EarthLocation,
) -> u.Quantity:
    """
    Compute the rotation speed of the FOV for a given evaluation time.

    Parameters
    ----------
    time_evaluation : astropy.time.Time
        The time at which the rotation speed should be evaluated.
    pointing_sky : astropy.coordinates.SkyCoord
        The direction pointed in the sky.
    observatory_earth_location : astropy.coordinates.EarthLocation
        The position of the observatory.

    Returns
    -------
    rotation_speed : astropy.units.Quantity
        The rotation speed of the FOV at the given time and pointing direction.
    """
    pointing_altaz = pointing_sky.transform_to(
        AltAz(obstime=time_evaluation, location=observatory_earth_location)
    )
    omega_earth = OMEGA_EARTH * u.rad
    omega = (
        omega_earth
        * np.cos(observatory_earth_location.lat)
        * np.cos(pointing_altaz.az)
        / np.cos(pointing_altaz.alt)
    )
    return omega


class PolarSkyOffsetFrame(SkyOffsetFrame, AltAz):
    # see https://github.com/astropy/astropy/pull/12845
    origin = CoordinateAttribute(
        frame=AltAz,
        default=None,
    )

    _default_representation = PhysicsSphericalRepresentation


@frame_transform_graph.transform(DynamicMatrixTransform, AltAz, PolarSkyOffsetFrame)
def reference_to_skyoffset_altaz_polar(reference_frame, skyoffset_frame):
    """Convert a reference coordinate to an sky offset frame."""

    # Define rotation matrices along the position angle vector, and
    # relative to the origin.

    origin = skyoffset_frame.origin.represent_as(PhysicsSphericalRepresentation)
    mat1 = rotation_matrix(-skyoffset_frame.rotation, "z")
    mat2 = rotation_matrix(origin.theta, "y")
    mat3 = rotation_matrix(origin.phi, "z")

    return mat1 @ mat2 @ mat3


@frame_transform_graph.transform(DynamicMatrixTransform, PolarSkyOffsetFrame, AltAz)
def skyoffset_to_reference_polar_altaz(skyoffset_coord, reference_frame):
    """Convert an sky offset frame coordinate to the reference frame"""

    # use the forward transform, but just invert it
    R = reference_to_skyoffset_altaz_polar(reference_frame, skyoffset_coord)
    # transpose is the inverse because R is a rotation matrix
    return matrix_transpose(R)


class EpsilonSkyOffsetFrame(SkyOffsetFrame, AltAz):
    origin = CoordinateAttribute(
        frame=AltAz,
        default=None,
    )

    _default_representation = PhysicsSphericalRepresentation

    frame_specific_representation_info = {
        PhysicsSphericalRepresentation: [
            RepresentationMapping("theta", "epsilon"),
            RepresentationMapping("phi", "phi"),
        ],
    }


@frame_transform_graph.transform(
    FunctionTransform, PolarSkyOffsetFrame, EpsilonSkyOffsetFrame
)
def reference_to_skyoffset_polar_eps(reference_frame, skyoffset_frame):
    """Convert a reference coordinate to an sky offset frame."""

    # Define rotation matrices along the position angle vector, and
    # relative to the origin.

    theta = reference_frame.theta
    phi = reference_frame.phi
    r = reference_frame.r
    sigma = 2
    epsilon = (
        np.exp(-(theta.deg**2) / sigma**2) * u.deg
    )  # it's not an angle anymore, anyways...
    # epsilon = np.pi * theta.deg**2 * u.deg
    # epsilon = theta.deg * u.deg

    representation = PhysicsSphericalRepresentation(phi=phi, theta=epsilon, r=r)

    return skyoffset_frame.realize_frame(representation)


@frame_transform_graph.transform(
    FunctionTransform, EpsilonSkyOffsetFrame, PolarSkyOffsetFrame
)
def reference_to_skyoffset_eps_polar(reference_frame, skyoffset_frame):
    """Convert a reference coordinate to an sky offset frame."""

    # Define rotation matrices along the position angle vector, and
    # relative to the origin.

    epsilon = reference_frame.epsilon
    phi = reference_frame.phi
    r = reference_frame.r

    # theta = np.sqrt(np.log(1/epsilon.deg)) * u.deg
    # theta = np.sqrt(epsilon.deg / np.pi) * u.deg
    # theta = epsilon.deg * u.deg
    # theta = np.sqrt(-np.log(epsilon.deg)) * u.deg
    EPS = 0
    sigma = 2
    theta = np.sqrt(-np.log(epsilon.deg + EPS) * sigma**2) * u.deg

    representation = PhysicsSphericalRepresentation(phi=phi, theta=theta, r=r)
    # print(representation)
    # print(skyoffset_frame.realize_frame(representation))

    return skyoffset_frame.realize_frame(representation)
