import numpy as np
import xarray as xr
import json
import os
import pandas as pd
from scipy.ndimage import convolve
from orcestra.utils import get_flight_segments

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to config.json relative to the script's location
config_path = os.path.join(script_dir, "config.json")

# read config.json file for parameters
with open(config_path) as f:
    config = json.load(f)


def _selective_where(ds, condition):
    """
    Apply where to dataset but do not touch georeference data which should not be masked

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    condition : xr.DataArray, np.array, or boolean
        The condition to apply to the dataset.

    Returns
    -------
    xr.Dataset
        The dataset with the condition applied.
    """

    ds = ds.assign(
        {
            var: ds[var].where(condition)
            for var in ds.data_vars
            if var not in config["plane_variables"]
        }
    )
    return ds


def _noise_filter_radar(ds):
    """Filter radar data by noise level.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered by noise level.

    """
    return ds.pipe(_selective_where, (ds.npw1 > config["noise_threshold"]))


def _state_filter_radar(ds):
    """Filter radar data for valid state.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered for valid state.
    """

    return ds.pipe(_selective_where, (ds.grst == config["valid_radar_state"]))


def _altitude_filter(ds):
    """Filter any dataset by plane altitude.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 dataset.

    Returns
    -------
    xr.Dataset
        Dataset filtered by plane altitude.
    """

    return ds.pipe(_selective_where, (ds.plane_altitude > config["altitude_threshold"]))


def _trim_dataset(ds, dim="time"):
    """
    Trim the dataset by removing data at the beginning and end until the first and last occurrence of valid data.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    dim : str
        The dimension along which to trim the dataset. Default is "time".

    Returns
    -------
    xr.Dataset
        The trimmed dataset.
    """
    # Drop NaNs along the specified dimension
    valid_data = ds.dropna(dim=dim, how="all")

    # Find the first and last indices of valid data
    first_valid_index = valid_data[dim].values[0]
    last_valid_index = valid_data[dim].values[-1]

    # Slice the dataset to include only the valid data
    trimmed_ds = ds.sel({dim: slice(first_valid_index, last_valid_index)})

    return trimmed_ds


def _filter_land(ds, sea_land_mask, offset=pd.Timedelta("7s")):
    """Filters out data that was collected over land.

    Removes data by offset earlier than the the time the plane flies over land
    to avoid including land measurements due to tilt of the plane or mask inaccuracies.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to filter.
    sea_land_mask : xr.DataArray
        Mask of land and sea. 1 for sea, 0 for land.
    offset : pd.Timedelta
        Time offset to remove data before the plane flies over land. Default is 7 seconds.

    Returns
    -------
    xr.Dataset
        Filtered dataset.
    """

    mask_path = sea_land_mask.sel(lat=ds.lat, lon=ds.lon, method="nearest")
    diff = (mask_path * 1).diff("time")
    start_land = diff.where(diff == -1).dropna("time").time
    end_land = diff.where(diff == 1).dropna("time").time
    for t in start_land:
        mask_path.loc[dict(time=slice(t - offset, t))] = 0
    for t in end_land:
        mask_path.loc[dict(time=slice(t, t + offset))] = 0
    return ds.pipe(_selective_where, (mask_path == 1).drop(["lat", "lon"]))


def _filter_clutter(ds):
    """Filter radar data for clutter.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered for clutter.
    """
    kernel = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    for var in [var for var in ds if ds[var].dims == ("time", "height")]:
        ds_binary = ~np.isnan(ds[var])
        neighbour_count = convolve(ds_binary, kernel, mode="constant", cval=False)
        clutter_mask = (ds_binary) & (~neighbour_count)
        ds = ds.assign({var: ds[var].where(~clutter_mask, np.nan)})
    return ds


def _add_clibration_mask(ds):
    """
    Add a mask for radar calibration segments to the radar dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.
    Returns
    -------
    xr.Dataset
        Radar dataset with a mask for radar calibration segments.
    """

    meta = get_flight_segments()
    radar_calib = [
        {**s, "platform_id": platform_id, "flight_id": flight_id}
        for platform_id, flights in meta.items()
        for flight_id, flight in flights.items()
        for s in flight["segments"]
        if "radar_calibration_wiggle" in s["kinds"]
    ]

    radar_calib_flight = [s for s in radar_calib if s["flight_id"] == ds.flight_id]
    mask_calib = xr.DataArray(
        np.zeros(ds.time.shape, dtype=bool),
        dims="time",
        coords={"time": ds.time},
        attrs={
            "long_name": "Mask for radar calibration segments",
            "description": "0 indicates no radar calibration, 1 indicates radar calibration",
        },
    )
    for s in radar_calib_flight:
        mask_calib.loc[{"time": slice(s["start"], s["end"])}] = True

    ds = ds.assign(mask_calibration=mask_calib)

    return ds


def _add_ground_mask(ds, sea_land_mask, threshold=40):
    """
    Add a mask for ground reflections to the radar dataset.

    The mask is created by identifying the first height level coming from the top for which dBZg exceeds the given threshold.
    All cells below this height level are considered to be ground reflections.

    Parameters
    ----------
    ds_radar : xr.Dataset
        Level1 radar dataset.
    sea_land_mask : xr.Dataset
        Dataset containing a mask for land and ocean. The mask should have the dimensions lat and lon.
        1 indicates ocean, 0 indicates land.
    threshold : int, optional
        dBZg threshold for ground reflection, by default 40.
    Returns
    -------
    xr.Dataset
        Radar dataset with a mask for ground reflections.
    """

    # get land or sea mask along track
    sea_mask = sea_land_mask.sel(lat=ds.lat, lon=ds.lon, method="nearest").drop_vars(
        ["lat", "lon"]
    )

    # get first height level coming from the top that exceeds 30 dbZg
    strong_signal = (ds.dBZg > threshold) * ds.height
    max_height = strong_signal.idxmax("height").interpolate_na(
        dim="time", method="linear"
    )

    # create mask for land and ocean
    dz = ds.height.diff("height").mean().values
    mask_land = ds.height <= (max_height + 6 * dz)
    mask_ocean = ds.height <= (max_height + 2 * dz)
    mask_ground_return = xr.where(sea_mask, mask_ocean, mask_land)

    # add mask to dataset
    ds = ds.assign(mask_ground_return=mask_ground_return)
    ds["mask_ground_return"].attrs = {
        "long_name": "Mask for ground reflections",
        "description": "1 indicates ground reflection, 0 indicates no ground reflection",
    }

    return ds


def _add_roll_mask(ds):
    """
    Add a mask for roll segments to the radar dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.
    Returns
    -------
    xr.Dataset
        Radar dataset with a mask for roll segments.
    """

    mask_roll = np.abs(ds.plane_roll) > 5
    mask_roll.attrs = {
        "long_name": "Mask for roll segments",
        "description": "1 indicates roll higher 5 deg, 0 indicates roll lower 5 deg",
    }
    ds = ds.assign(mask_roll=mask_roll)

    return ds


def add_metadata_radar(ds, flight_id):
    """
    Add metadata to the radar dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.
    Returns
    -------
    xr.Dataset
        Radar dataset with metadata.
    """

    # add new attrs
    ds.attrs["flight_id"] = flight_id
    ds.attrs["title"] = "MIRA Cloud Radar Data Moments"
    ds.attrs["summary"] = (
        "Processed data from the MIRA cloud radar onboard the HALO aircraft"
    )
    ds.attrs["creator_name"] = "Jakob Deutloff, Lukas Kluft"
    ds.attrs["creator_email"] = (
        "jakob.deutloff@uni-hamburg.de, lukas.kluft@mpimet.mpg.de"
    )
    ds.attrs["project"] = "ORCESTRA, PERCUSION"
    ds.attrs["platform"] = "HALO"
    ds.attrs["history"] = (
        "The processing software is available at https://github.com/orcestra-campaign/hamp_processing"
    )
    ds.attrs["licence"] = "CC-BY-4.0"
    ds.attrs["featureType"] = "trajectory" / "trajectoryProfile"
    ds.attrs["references"] = "10.5194/amt-12-1815-2019, 10.5194/essd-13-5545-2021"
    ds.attrs["keywords"] = "Cloud Radar, HALO, ORCESTRA, PERCUSION, Tropical Atlantic"

    # remove oudated attrs
    ds.attrs.pop("Copywright", None)
    ds.attrs.pop("Copywright_Owner", None)
    ds.attrs.pop("Latitude", None)
    ds.attrs.pop("Longitude", None)
    ds.attrs.pop("Altitude", None)

    return ds


def filter_radar(ds):
    """Filter radar data for noise, valid radar states, and roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data filtered for noise, state, and roll angle.
    """

    return (
        ds.pipe(_noise_filter_radar)
        .pipe(_state_filter_radar)
        .pipe(_trim_dataset)
        .pipe(_filter_clutter)
    )


def filter_radiometer(ds, sea_land_mask):
    """Filter radiometer data for height and roll angle.

    Parameters
    ----------
    ds : xr.Dataset
        Level0 radiometer dataset.
    height : xr.DataArray
        Flight altitude in m.
    roll : xr.DataArray
        Flight roll angle in degrees.
    lat : xr.DataArray
        Latitudes of the flightpath.
    lon : xr.DataArray
        Longitudes of the flightpath.
    sea_land_mask : xr.DataArray
        Mask of land and sea. 1 for sea, 0 for land.

    Returns
    -------
    xr.Dataset
        Radiometer data filtered for height and roll angle.
    """

    return (
        ds.pipe(_altitude_filter).pipe(_trim_dataset).pipe(_filter_land, sea_land_mask)
    )


def add_masks_radar(ds, sea_land_mask):
    """Add masks to radar data for calibration, ground reflections, and roll segments.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.
    sea_land_mask : xr.DataArray
        Mask of land and sea. 1 for sea, 0 for land.

    Returns
    -------
    xr.Dataset
        Radar dataset with masks for calibration, ground reflections, and roll segments.
    """

    return (
        ds.pipe(_add_clibration_mask)
        .pipe(_add_ground_mask, sea_land_mask)
        .pipe(_add_roll_mask)
    )


def filter_spikes(ds, threshold=5, window=1200):
    """
    Filters out spikes in a time series by comparing the difference between each point
    and the minimum of the surrounding 5 minutes.

    Parameters
    ----------
    ds : xr.DataArray
        DataArray to filter.
    threshold : float
        Maximum allowed difference between the data and the minimum within the window.
    window : int
        Size of the window in seconds to compare the data, default is 5 minutes.

    Returns
    -------
    xr.DataArray
        Filtered DataArray.
    """
    diff = ds - ds.rolling(time=window, center=True, min_periods=1).min()
    filtered = ds.where(abs(diff) < threshold)
    interpolated = filtered.interpolate_na("time", method="linear")
    return xr.where(abs(diff) < threshold, ds, interpolated)


def correct_radar_height(ds):
    """Correct radar range gates with HALO flight altitude to height above WGS84 ellipsoid.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radar dataset.

    Returns
    -------
    xr.Dataset
        Radar data corrected to height above WGS84 ellipsoid.

    """
    z_grid = np.arange(0, ds.plane_altitude.max() + 30, 30)

    flight_los = (
        ds.plane_altitude
        / np.cos(np.radians(ds.plane_pitch))
        / np.cos(np.radians(ds.plane_roll))
    )

    ds_z_grid = xr.DataArray(
        data=np.tile(z_grid, (len(ds.time), 1)),
        dims=("time", "height"),
        coords={"time": ds.time, "height": z_grid},
    )
    ds_range = xr.DataArray(
        coords={"time": ds.time, "height": z_grid},
        dims=("time", "height"),
        data=ds_z_grid
        / np.cos(np.radians(ds.plane_pitch))
        / np.cos(np.radians(ds.plane_roll)),
    )

    ds = ds.sel(range=flight_los - ds_range, method="nearest").drop_vars("range")
    ds = ds.assign(
        {
            var: ds[var].where(ds_z_grid < ds.plane_altitude)
            for var in ds
            if ds[var].dims == ("time", "height")
        }
    )
    return ds
