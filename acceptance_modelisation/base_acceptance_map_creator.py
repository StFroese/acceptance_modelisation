import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional

import astropy.units as u
from matplotlib import ExecutableNotFoundError
import numpy as np
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time
from gammapy.data import Observations, Observation
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D, Background3D
from gammapy.irf.background import BackgroundIRF
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.maps import WcsNDMap, WcsGeom, Map, MapAxis
from regions import CircleSkyRegion, SkyRegion
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import (
    interp1d,
    NearestNDInterpolator,
    LinearNDInterpolator,
    CloughTocher2DInterpolator,
)
from scipy.ndimage import gaussian_filter

from .toolbox import (
    compute_rotation_speed_fov,
    PolarSkyOffsetFrame,
    EpsilonSkyOffsetFrame,
)


class BaseAcceptanceMapCreator(ABC):
    def __init__(
        self,
        energy_axis: MapAxis,
        max_offset: u.Quantity,
        spatial_resolution: u.Quantity,
        exclude_regions: Optional[List[SkyRegion]] = None,
        min_observation_per_cos_zenith_bin: int = 3,
        initial_cos_zenith_binning: float = 0.01,
        max_fraction_pixel_rotation_fov: float = 0.5,
        time_resolution_rotation_fov: u.Quantity = 0.1 * u.s,
        polar: bool = False,
    ) -> None:
        """
        Create the class for calculating radial acceptance model.

        Parameters
        ----------
        energy_axis : gammapy.maps.geom.MapAxis
            The energy axis for the acceptance model
        max_offset : astropy.units.Quantity
            The offset corresponding to the edge of the model
        spatial_resolution : astropy.units.Quantity
            The spatial resolution
        exclude_regions : list of regions.SkyRegion, optional
            Regions with known or putative gamma-ray emission, will be excluded from the calculation of the acceptance map
        min_observation_per_cos_zenith_bin : int, optional
            Minimum number of observations per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution_rotation_fov : astropy.units.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV
        polar: bool, optional
            Generate the acceptance map in polar coordinates like MAGIC (exp(-R^2), phi)
        """

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

        # Store base parameter
        self.energy_axis = energy_axis
        self.max_offset = max_offset
        self.exclude_regions = exclude_regions
        self.min_observation_per_cos_zenith_bin = min_observation_per_cos_zenith_bin
        self.initial_cos_zenith_binning = initial_cos_zenith_binning

        # Calculate map parameter
        self.n_bins_map = 2 * int(
            np.rint((self.max_offset / spatial_resolution).to(u.dimensionless_unscaled))
        )
        self.spatial_bin_size = self.max_offset / (self.n_bins_map / 2)
        self.center_map = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
        self.geom = WcsGeom.create(
            skydir=self.center_map,
            npix=(self.n_bins_map, self.n_bins_map),
            binsz=self.spatial_bin_size,
            frame="icrs",
            axes=[self.energy_axis],
        )
        if polar:
            self.geom = WcsGeom.create(
                skydir=SkyCoord(ra=0 * u.deg, dec=0 * u.deg, frame="icrs"),
                binsz=(30, 0.4),
                width=(360, 28),
                frame="icrs",
                axes=[self.energy_axis],
                refpix=(6.49, 0),
            )
            self.geom_grid = WcsGeom.create(
                skydir=self.center_map,
                npix=(self.n_bins_map, self.n_bins_map),
                binsz=self.spatial_bin_size,
                frame="icrs",
                axes=[self.energy_axis],
            )
        logging.info(
            "Computation will be made with a bin size of {:.3f} arcmin".format(
                self.spatial_bin_size.to_value(u.arcmin)
            )
        )

        # Store rotation computation parameters
        self.max_fraction_pixel_rotation_fov = max_fraction_pixel_rotation_fov
        self.time_resolution_rotation_fov = time_resolution_rotation_fov

        self.polar = polar

    def _transform_obs_to_camera_frame(
        self, obs: Observation
    ) -> Tuple[Observation, List[SkyRegion]]:
        """
        Transform events, pointing and exclusion regions of an obs from a sky frame to camera frame

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation to transform

        Returns
        -------
        obs_camera_frame : gammapy.data.observations.Observation
            The observation transformed for reference in camera frame
        exclusion_region_camera_frame : list of region.SkyRegion
            The list of exclusion region in camera frame
        """

        # Transform to altaz frame
        altaz_frame = AltAz(obstime=obs.events.time, location=obs.meta.location)
        events_altaz = obs.events.radec.transform_to(altaz_frame)
        pointing_altaz = obs.get_pointing_icrs(obs.tmid).transform_to(altaz_frame)

        # Rotation to transform to camera frame
        camera_frame = SkyOffsetFrame(
            origin=AltAz(
                alt=pointing_altaz.alt,
                az=pointing_altaz.az,
                obstime=obs.events.time,
                location=obs.meta.location,
            ),
            rotation=[
                0.0,
            ]
            * len(obs.events.time)
            * u.deg,
        )

        if self.polar:
            polar_frame = PolarSkyOffsetFrame(origin=camera_frame.origin)
            events_camera_polar_frame = events_altaz.transform_to(polar_frame)

            epsilon_frame = EpsilonSkyOffsetFrame(origin=polar_frame.origin)
            events_camera_frame = events_camera_polar_frame.transform_to(epsilon_frame)
        else:
            events_camera_frame = events_altaz.transform_to(camera_frame)

        # Formatting data for the output
        camera_frame_events = obs.events.copy()
        camera_frame_events.table["RA"] = (
            events_camera_frame.lon if not self.polar else events_camera_frame.phi
        )
        camera_frame_events.table["DEC"] = (
            events_camera_frame.lat if not self.polar else events_camera_frame.epsilon
        )
        camera_frame_obs_info = copy.deepcopy(obs.obs_info)
        camera_frame_obs_info["RA_PNT"] = 0.0 if not self.polar else 0
        camera_frame_obs_info["DEC_PNT"] = (
            0.0 if not self.polar else 0
        )  # solve sqrt(pi)/2 erf(x) = 0.5

        obs_camera_frame = Observation(
            obs_id=obs.obs_id,
            obs_info=camera_frame_obs_info,
            events=camera_frame_events,
            gti=obs.gti,
            aeff=obs.aeff,
        )
        obs_camera_frame._location = obs.meta.location

        # Compute the exclusion region in camera frame for the average time
        average_alt_az_frame = AltAz(
            obstime=obs_camera_frame.tmid,
            location=obs_camera_frame.meta.location,
        )
        average_alt_az_pointing = obs.get_pointing_icrs(obs.tmid).transform_to(
            average_alt_az_frame
        )
        exclusion_region_camera_frame = (
            self._transform_exclusion_region_to_camera_frame(average_alt_az_pointing)
        )

        return obs_camera_frame, exclusion_region_camera_frame

    def _transform_exclusion_region_to_camera_frame(
        self, pointing_altaz: AltAz
    ) -> List[SkyRegion]:
        """
        Transform the list of exclusion regions in sky frame into a list in camera frame.

        Parameters
        ----------
        pointing_altaz : astropy.coordinates.AltAz
            The pointing position of the telescope.

        Returns
        -------
        exclusion_region_camera_frame : list of regions.SkyRegion
            The list of exclusion regions in camera frame.

        Raises
        ------
        Exception
            If the region type is not supported.
        """

        camera_frame = SkyOffsetFrame(
            origin=pointing_altaz,
            rotation=[
                0.0,
            ]
            * u.deg,
        )
        exclude_region_camera_frame = []
        for region in self.exclude_regions:
            if isinstance(region, CircleSkyRegion):
                center_coordinate = region.center
                center_coordinate_altaz = center_coordinate.transform_to(pointing_altaz)
                center_coordinate_camera_frame = center_coordinate_altaz.transform_to(
                    camera_frame
                )
                center_coordinate_camera_frame_arb = SkyCoord(
                    ra=center_coordinate_camera_frame.lon[0],
                    dec=center_coordinate_camera_frame.lat[0],
                )
                exclude_region_camera_frame.append(
                    CircleSkyRegion(
                        center=center_coordinate_camera_frame_arb, radius=region.radius
                    )
                )
            else:
                raise Exception(f"{type(region)} region type not supported")

        return exclude_region_camera_frame

    def _create_map(
        self,
        obs: Observation,
        geom: WcsGeom,
        exclude_regions: List[SkyRegion],
        add_bkg: bool = False,
    ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create a map and the associated exclusion mask based on the given geometry and exclusion region.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        geom : gammapy.maps.WcsGeom
            The geometry for the maps.
        exclude_regions : list of regions.SkyRegion
            The list of exclusion regions.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        maker = MapDatasetMaker(selection=["counts"])
        if add_bkg:
            maker = MapDatasetMaker(selection=["counts", "background"])

        # maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=self.max_offset)
        if self.polar:
            geom_image = self.geom_grid.to_image()
        else:
            geom_image = geom.to_image()
        exclusion_mask = (
            ~geom_image.region_mask(exclude_regions)
            if len(exclude_regions) > 0
            else ~Map.from_geom(geom_image)
        )

        map_obs = maker.run(MapDataset.create(geom=geom), obs)
        # map_obs = maker_safe_mask.run(map_obs, obs)

        return map_obs, exclusion_mask

    def _create_sky_map(
        self, obs: Observation, add_bkg: bool = False
    ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create the sky map and the associated exclusion mask based on the observation and the exclusion regions.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        geom_obs = WcsGeom.create(
            skydir=obs.get_pointing_icrs(obs.tmid),
            npix=(self.n_bins_map, self.n_bins_map),
            binsz=self.spatial_bin_size,
            frame="icrs",
            axes=[self.energy_axis],
        )
        map_obs, exclusion_mask = self._create_map(
            obs, geom_obs, self.exclude_regions, add_bkg=add_bkg
        )

        return map_obs, exclusion_mask

    def _create_camera_map(self, obs: Observation) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create the sky map in camera coordinates used for computation.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset in camera coordinates.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask in camera coordinates.
        """

        obs_camera_frame, exclusion_region_camera_frame = (
            self._transform_obs_to_camera_frame(obs)
        )
        map_obs, exclusion_mask = self._create_map(
            obs_camera_frame, self.geom, exclusion_region_camera_frame, add_bkg=False
        )

        return map_obs, exclusion_mask

    def _compute_time_intervals_based_on_fov_rotation(self, obs: Observation) -> Time:
        """
        Calculate time intervals based on the rotation of the Field of View (FoV).

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to calculate time intervals.

        Returns
        -------
        time_intervals : astropy.time.Time
            The time intervals for cutting the observation into time bins.
        """

        # Determine time interval for cutting the obs as function of the rotation of the Fov
        n_bin = max(
            2,
            int(
                np.rint(
                    (
                        (obs.tstop - obs.tstart) / self.time_resolution_rotation_fov
                    ).to_value(u.dimensionless_unscaled)
                )
            ),
        )
        time_axis = np.linspace(obs.tstart, obs.tstop, num=n_bin)
        rotation_speed_fov = compute_rotation_speed_fov(
            time_axis, obs.get_pointing_icrs(obs.tmid), obs.meta.location
        )
        rotation_fov = (
            cumulative_trapezoid(
                x=time_axis.unix_tai,
                y=rotation_speed_fov.to_value(u.rad / u.s),
                initial=0.0,
            )
            * u.rad
        )
        distance_rotation_fov = rotation_fov.to_value(u.rad) * np.pi * self.max_offset
        node_obs = distance_rotation_fov // (
            self.spatial_bin_size * self.max_fraction_pixel_rotation_fov
        )
        change_node = node_obs[2:] != node_obs[1:-1]
        time_interval = Time(
            [
                obs.tstart,
            ]
            + [
                time_axis[1:-1][change_node],
            ]
            + [
                obs.tstop,
            ]
        )

        return time_interval

    def _create_base_computation_map(
        self, observations: Observation
    ) -> Tuple[WcsNDMap, WcsNDMap, WcsNDMap, u.Unit]:
        """
        From a list of observations return a stacked finely binned counts and exposure map in camera frame to compute a model

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The list of observations

        Returns
        -------
        count_map_background : gammapy.map.WcsNDMap
            The count map
        exp_map_background : gammapy.map.WcsNDMap
            The exposure map corrected for exclusion regions
        exp_map_background_total : gammapy.map.WcsNDMap
            The exposure map without correction for exclusion regions
        livetime : astropy.unit.Unit
            The total exposure time for the model
        """
        if self.polar:
            count_map_background = WcsNDMap(geom=self.geom_grid)
            exp_map_background = WcsNDMap(geom=self.geom_grid, unit=u.s)
            exp_map_background_total = WcsNDMap(geom=self.geom_grid, unit=u.s)
        else:
            count_map_background = WcsNDMap(geom=self.geom)
            exp_map_background = WcsNDMap(geom=self.geom, unit=u.s)
            exp_map_background_total = WcsNDMap(geom=self.geom, unit=u.s)

        livetime = 0.0 * u.s

        with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
            for obs in observations:
                time_interval = self._compute_time_intervals_based_on_fov_rotation(obs)
                for i in range(len(time_interval) - 1):
                    cut_obs = obs.select_time(
                        Time([time_interval[i], time_interval[i + 1]])
                    )
                    count_map_obs, exclusion_mask = self._create_camera_map(cut_obs)

                    if self.polar:
                        map_coords = count_map_obs.counts.geom.to_image().get_coord()
                        pointing_altaz = obs.get_pointing_altaz(Time(time_interval[i]))
                        coords_eps = SkyCoord(
                            epsilon=map_coords.lat,
                            phi=map_coords.lon,
                            r=1,
                            frame=EpsilonSkyOffsetFrame(origin=pointing_altaz),
                        )

                        polar_frame = PolarSkyOffsetFrame(origin=pointing_altaz)
                        coords_polar = coords_eps.transform_to(polar_frame)

                        sky_offset_frame = SkyOffsetFrame(origin=pointing_altaz)
                        coords_altaz = coords_polar.transform_to(sky_offset_frame)
                        lon = coords_altaz.lon.to_value("deg").ravel()
                        lat = coords_altaz.lat.to_value("deg").ravel()

                        map_coords_interp = self.geom_grid.to_image().get_coord()
                        skycoord_interp = map_coords_interp.skycoord
                        sky_offset_frame_interp = SkyOffsetFrame(
                            origin=SkyCoord("0 deg", "0 deg", frame="icrs")
                        )
                        skycoord_interp = skycoord_interp.transform_to(
                            sky_offset_frame_interp
                        )

                        lon_interp = skycoord_interp.lon.to_value("deg")
                        lat_interp = skycoord_interp.lat.to_value("deg")

                        gridded_data = []
                        for energy_bin_data in count_map_obs.counts.data:
                            # print(np.sum(energy_bin_data))
                            # import matplotlib.pyplot as plt

                            # plt.imshow(energy_bin_data)
                            # plt.show()
                            # print(energy_bin_data.shape)

                            # energy_bin_data = gaussian_filter(
                            #     energy_bin_data, sigma=1, mode="wrap", axes=1
                            # )
                            # plt.imshow(energy_bin_data)
                            # plt.show()
                            # energy_bin_data = gaussian_filter(
                            #     energy_bin_data, sigma=1, mode="reflect", axes=0
                            # )
                            # plt.imshow(energy_bin_data)
                            # plt.show()
                            # import matplotlib.pyplot as plt
                            #
                            # plt.imshow(energy_bin_data)
                            # plt.show()
                            #
                            # plt.plot(lon, lat, "x")
                            # plt.show()
                            interpolator = NearestNDInterpolator(
                                list(zip(lon, lat)),
                                energy_bin_data.ravel(),
                                # fill_value=0.0,
                            )

                            interp_data = interpolator(lon_interp, lat_interp)

                            norm_before = np.sum(energy_bin_data)
                            norm_now = np.sum(interp_data)
                            if (norm_before != np.nan and norm_before != 0) and (
                                norm_now != np.nan and norm_now != 0
                            ):
                                interp_data = interp_data / norm_now * norm_before

                            # print(np.sum(interp_data[interp_data != np.nan]))
                            # print(norm)
                            # import matplotlib.pyplot as plt
                            #
                            # plt.plot(lon, lat, "x")
                            # plt.show()
                            # plt.plot(lon_interp, lat_interp, "x")
                            # plt.show()

                            #
                            # plt.imshow(energy_bin_data)
                            # plt.show()
                            # plt.imshow(interp_data)
                            # plt.show()

                            # print(np.sum(interp_data))
                            #
                            # smeared_data = gaussian_filter(
                            #     interp_data, sigma=10, mode="reflect"
                            # )
                            # print(np.sum(smeared_data))
                            smeared_data = interp_data
                            # #
                            # import matplotlib.pyplot as plt
                            #
                            # plt.imshow(interp_data)
                            # plt.show()
                            # plt.imshow(smeared_data)
                            # plt.show()

                            smeared_data[lon_interp**2 + lat_interp**2 > 9] = 0
                            gridded_data.append(smeared_data)
                            # import matplotlib.pyplot as plt
                            #
                            # plt.imshow(smeared_data)
                            # plt.show()
                            # gridded_data.append(interp_data)

                    if self.polar:
                        exp_map_obs = MapDataset.create(geom=self.geom_grid)
                        exp_map_obs_total = MapDataset.create(geom=self.geom_grid)
                    else:
                        exp_map_obs = MapDataset.create(
                            geom=count_map_obs.geoms["geom"]
                        )
                        exp_map_obs_total = MapDataset.create(
                            geom=count_map_obs.geoms["geom"]
                        )
                    exp_map_obs.counts.data = (
                        cut_obs.observation_live_time_duration.value
                    )
                    exp_map_obs_total.counts.data = (
                        cut_obs.observation_live_time_duration.value
                    )

                    if self.polar:
                        # exp_data_total, _, _ = np.histogram2d(lon, lat, bins=self.n_bins_map, weights=cut_obs.observation_live_time_duration.value*np.ones_like(lon))

                        # interpolator = NearestNDInterpolator(
                        #     list(zip(lon, lat)),
                        #     cut_obs.observation_live_time_duration.value
                        #     * np.ones_like(lon),
                        # )
                        # exp_data_total = interpolator(lon_interp, lat_interp)
                        # import matplotlib.pyplot as plt
                        #
                        # plt.imshow(exp_data_total)
                        # plt.show()

                        # exp_data = copy.deepcopy(exp_data_total)

                        for j in range(len(gridded_data)):
                            gridded_data[j][:, :] = (
                                gridded_data[j][:, :] * exclusion_mask
                            )
                            exp_map_obs.counts.data[j, :, :] = (
                                exp_map_obs.counts.data[j, :, :] * exclusion_mask
                            )
                            # exp_map_obs.counts.data[j, :, :] = gaussian_filter(
                            #     exp_map_obs.counts.data[j, :, :],
                            #     sigma=10,
                            #     mode="reflect",
                            # )
                        # exp_data = exp_data * exclusion_mask

                        count_map_background.data += gridded_data
                        # exp_map_background.data += exp_data[None, ...]
                        exp_map_background.data += exp_map_obs.counts.data
                        exp_map_background_total.data += exp_map_obs_total.counts.data
                        # exp_map_background_total.data += exp_data_total
                        livetime += cut_obs.observation_live_time_duration

                    else:
                        for j in range(count_map_obs.counts.data.shape[0]):
                            count_map_obs.counts.data[j, :, :] = (
                                count_map_obs.counts.data[j, :, :] * exclusion_mask
                            )
                            exp_map_obs.counts.data[j, :, :] = (
                                exp_map_obs.counts.data[j, :, :] * exclusion_mask
                            )

                        count_map_background.data += count_map_obs.counts.data
                        exp_map_background.data += exp_map_obs.counts.data
                        exp_map_background_total.data += exp_map_obs_total.counts.data
                        livetime += cut_obs.observation_live_time_duration
                # import matplotlib.pyplot as plt
                #
                # plt.imshow(count_map_background.data[0])
                # plt.show()
                # plt.imshow(
                #     exp_map_background.data[0] / exp_map_background_total.data[0]
                # )
                # plt.imshow(
                #     count_map_background.data[0]
                #     / exp_map_background.data[0]
                #     * exp_map_background_total.data[0]
                # )
                # plt.show()

        return (
            count_map_background,
            exp_map_background,
            exp_map_background_total,
            livetime,
        )

    @abstractmethod
    def create_acceptance_map(self, observations: Observations) -> BackgroundIRF:
        """
        Abstract method to calculate an acceptance map from a list of observations.

        Subclasses must implement this method to provide the specific algorithm for calculating the acceptance map.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to create the acceptance map.

        Returns
        -------
        acceptance_map : gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            The acceptance map calculated using the specific algorithm implemented by the subclass.
        """
        pass

    def _normalised_model_per_run(
        self, observations: Observations, acceptance_map: dict[Any, BackgroundIRF]
    ) -> dict[Any, BackgroundIRF]:
        """
        Normalised the acceptance model associated to each run to the events associated with the run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        acceptance_map :dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
            This is the models that will be normalised
        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
        """

        normalised_acceptance_map = {}
        # Fit norm of the model to the observations
        for obs in observations:
            id_observation = obs.obs_id

            # replace the background model
            modified_observation = copy.deepcopy(obs)
            modified_observation.bkg = acceptance_map[id_observation]

            # Fit the background model
            logging.info("Fit to model to run " + str(id_observation))
            map_obs, exclusion_mask = self._create_sky_map(
                modified_observation, add_bkg=True
            )
            maker_FoV_background = FoVBackgroundMaker(
                method="fit", exclusion_mask=exclusion_mask
            )
            map_obs = maker_FoV_background.run(map_obs)

            # Extract the normalisation
            parameters = map_obs.models.to_parameters_table()
            norm_background = parameters[parameters["name"] == "norm"]["value"][0]

            if norm_background < 0.0:
                logging.error(
                    "Invalid normalisation value for run "
                    + str(id_observation)
                    + " : "
                    + str(norm_background)
                )
            elif norm_background > 1.5 or norm_background < 0.5:
                logging.warning(
                    "High correction of the background normalisation normalisation for run "
                    + str(id_observation)
                    + " : "
                    + str(norm_background)
                )

            # Apply normalisation to the background model
            normalised_acceptance_map[id_observation] = copy.deepcopy(
                acceptance_map[id_observation]
            )
            normalised_acceptance_map[id_observation].data = (
                normalised_acceptance_map[id_observation].data * norm_background
            )

        return normalised_acceptance_map

    def create_acceptance_map_cos_zenith_binned(
        self, observations: Observations
    ) -> dict[Any, BackgroundIRF]:
        """
        Calculate an acceptance map using cos zenith binning and interpolation

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key

        """

        cos_zenith_bin = np.sort(
            np.arange(
                1.0,
                0.0 - self.initial_cos_zenith_binning,
                -self.initial_cos_zenith_binning,
            )
        )
        cos_zenith_observations = [
            np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in observations
        ]
        run_per_bin = np.histogram(cos_zenith_observations, bins=cos_zenith_bin)[0]

        i = 0
        while i < len(run_per_bin):
            if run_per_bin[i] < self.min_observation_per_cos_zenith_bin and (
                i + 1
            ) < len(run_per_bin):
                run_per_bin[i] += run_per_bin[i + 1]
                run_per_bin = np.delete(run_per_bin, i + 1)
                cos_zenith_bin = np.delete(cos_zenith_bin, i + 1)
            elif (
                run_per_bin[i] < self.min_observation_per_cos_zenith_bin
                and (i + 1) == len(run_per_bin)
                and i > 0
            ):
                run_per_bin[i - 1] += run_per_bin[i]
                run_per_bin = np.delete(run_per_bin, i)
                cos_zenith_bin = np.delete(cos_zenith_bin, i)
                i -= 1
            else:
                i += 1

        binned_observations = []
        for i in range((len(cos_zenith_bin) - 1)):
            binned_observations.append(Observations())
        for obs in observations:
            binned_observations[
                np.digitize(
                    np.cos(obs.get_pointing_altaz(obs.tmid).zen), cos_zenith_bin
                )
                - 1
            ].append(obs)

        binned_model = [
            self.create_acceptance_map(binned_obs) for binned_obs in binned_observations
        ]
        bin_center = []
        for i in range(len(binned_observations)):
            weighted_cos_zenith_bin_per_obs = []
            livetime_per_obs = []
            for obs in binned_observations[i]:
                weighted_cos_zenith_bin_per_obs.append(
                    obs.observation_live_time_duration
                    * np.cos(obs.get_pointing_altaz(obs.tmid).zen)
                )
                livetime_per_obs.append(obs.observation_live_time_duration)
            bin_center.append(
                np.sum([wcos.value for wcos in weighted_cos_zenith_bin_per_obs])
                / np.sum([livet.value for livet in livetime_per_obs])
            )

        acceptance_map = {}
        if len(binned_model) <= 1:
            logging.warning("Only one zenith bin, zenith interpolation deactivated")
            for obs in observations:
                acceptance_map[obs.obs_id] = binned_model[0]
        else:
            data_cube = (
                np.zeros(
                    tuple(
                        [
                            len(binned_model),
                        ]
                        + list(binned_model[0].data.shape)
                    )
                )
                * binned_model[0].unit
            )
            for i in range(len(binned_model)):
                data_cube[i] = binned_model[i].data * binned_model[i].unit
            interp_func = interp1d(
                x=np.array(bin_center),
                y=np.log10(data_cube.value + np.finfo(np.float64).tiny),
                axis=0,
                fill_value="extrapolate",
            )
            for obs in observations:
                data_obs = 10.0 ** interp_func(
                    np.cos(obs.get_pointing_altaz(obs.tmid).zen)
                )
                data_obs[data_obs < 100 * np.finfo(np.float64).tiny] = 0.0
                if type(binned_model[0]) is Background2D:
                    acceptance_map[obs.obs_id] = Background2D(
                        axes=binned_model[0].axes, data=data_obs * data_cube.unit
                    )
                elif type(binned_model[0]) is Background3D:
                    acceptance_map[obs.obs_id] = Background3D(
                        axes=binned_model[0].axes, data=data_obs * data_cube.unit
                    )
                else:
                    raise Exception("Unknown background format")

        return acceptance_map

    def create_acceptance_map_per_observation(
        self,
        observations: Observations,
        zenith_bin: bool = True,
        runwise_normalisation: bool = True,
    ) -> dict[Any, BackgroundIRF]:
        """
        Calculate an acceptance map with the norm adjusted for each run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        zenith_bin : bool, optional
            If true the acceptance maps will be generated using zenith binning and interpolation
        runwise_normalisation : bool, optional
            If true the acceptance maps will be normalised runwise to the observations

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
        """

        acceptance_map = {}
        if zenith_bin:
            acceptance_map = self.create_acceptance_map_cos_zenith_binned(observations)
        else:
            unique_base_acceptance_map = self.create_acceptance_map(observations)
            for obs in observations:
                acceptance_map[obs.obs_id] = unique_base_acceptance_map

        if runwise_normalisation:
            acceptance_map = self._normalised_model_per_run(
                observations, acceptance_map
            )

        return acceptance_map
