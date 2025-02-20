import numpy as np
import xarray as xr
import json
import os
import pandas as pd
from scipy.ndimage import convolve
from orcestra.utils import get_flight_segments
import yaml

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


def _bahamas_fix_time(ds):
    """Fix time coordinate of BAHAMAS datasets."""
    return ds.rename({"tid": "time", "TIME": "time"}).set_index(time="time")


def _radar_fix_time(ds):
    """Fix time coordinate of RADAR moments datasets."""
    datetime = (
        np.datetime64("1970-01-01", "ns")
        + ds.time.values * np.timedelta64(1, "s")
        + ds.microsec.values * np.timedelta64(1, "us")
    )

    return ds.assign(time=datetime).drop_vars("microsec")


def _radar_add_dBZ(ds):
    """Add reflectivity in dB."""
    ds = ds.assign(
        dBZg=lambda dx: 10 * np.log10(dx.Zg),
        dBZe=lambda dx: 10 * np.log10(dx.Ze),
    )
    ds.dBZg.attrs = {
        "units": "dBZg",
        "long_name": "Decadal logarithm of equivalent radar reflectivity of all targets (Zg)",
    }
    ds.dBZe.attrs = {
        "units": "dBZe",
        "long_name": "Decadal logarithm of equivalent radar reflectivity of hydrometeors (Ze)",
    }

    return ds


def _fix_radiometer_time(ds):
    """Replace duplicates in time coordinate of radiometer datasets with correct time and ensure 4Hz frequency."""

    # replace duplicate values
    time_broken = ds.time.values
    first_occurence = time_broken[0]
    n = 0
    time_new = []
    for i in range(len(time_broken)):
        if time_broken[i] == first_occurence:
            time_new.append(time_broken[i] + pd.Timedelta("0.25s") * n)
            n += 1
        else:
            n = 1
            first_occurence = time_broken[i]
            time_new.append(first_occurence)

    ds = ds.assign_coords(time=time_new).sortby("time").drop_duplicates("time")

    # ensure 4Hz frequency
    start_time = ds.time.min().values
    end_time = ds.time.max().values
    time_expected = pd.date_range(start=start_time, end=end_time, freq="0.25s")
    ds = ds.reindex(time=time_expected, fill_value=np.nan)

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


def _add_sea_land_mask(ds, sea_land_mask, offset=pd.Timedelta("7s")):
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
    ds = ds.assign(mask_sea_land=mask_path.drop_vars(["lat", "lon"]).astype(int))
    ds["mask_sea_land"].attrs = {
        "long_name": "Mask for sea and land.",
        "description": "1 indicates sea, 0 indicates land.",
    }
    return ds


def _add_amplifier_fault_mask(ds):
    """Add a mask for amplifier faults to the radiometer dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset.
    Returns
    -------
    xr.Dataset
        Radiometer dataset with a mask for amplifier faults.
    """

    date = pd.Timestamp(ds.time.values[0]).strftime("%Y%m%d")
    path = f"{config['path_error_file']}HALO-{date}.yaml"
    if os.path.exists(path):
        with open(path) as f:
            amplifier_faults = yaml.safe_load(f)
    else:
        amplifier_faults = {}

    mask_amplifier_fault = xr.DataArray(
        np.ones(ds.TBs.shape),
        dims=ds.TBs.dims,
        coords=ds.TBs.coords,
        attrs={
            "long_name": "Mask for amplifier faults",
            "description": "1 indicates no amplifier fault, 0 indicates amplifier fault",
        },
    )

    freq_of_module = {
        "KV": slice(22, 58),
        "183": slice(183, 191),
        "119": slice(120, 128),
        "90": 90,
    }

    for module in amplifier_faults.keys():
        for time_tuple in amplifier_faults[module]:
            mask_amplifier_fault.loc[
                {
                    "time": slice(f"{date}T{time_tuple[0]}", f"{date}T{time_tuple[1]}"),
                    "frequency": freq_of_module[module],
                }
            ] = 0

    ds = ds.assign(mask_amplifier_fault=mask_amplifier_fault.astype(int))

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
        np.ones(ds.time.shape),
        dims="time",
        coords={"time": ds.time},
        attrs={
            "long_name": "Mask for radar calibration segments",
            "description": "1 indicates no radar calibration, 0 indicates radar calibration",
        },
    )
    for s in radar_calib_flight:
        mask_calib.loc[{"time": slice(s["start"], s["end"])}] = 0

    ds = ds.assign(mask_calibration=mask_calib.astype(int))

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

    # get first height level coming from the top that exceeds threshold
    strong_signal = (ds.dBZg > threshold) * ds.height
    max_height = strong_signal.idxmax("height").interpolate_na(
        dim="time", method="linear"
    )

    # create mask for land and ocean
    dz = ds.height.diff("height").mean().values
    mask_land = ds.height > (max_height + 6 * dz)
    mask_ocean = ds.height > (max_height + 2 * dz)
    mask_ground_return = xr.where(sea_mask, mask_ocean, mask_land).astype(int)

    # add mask to dataset
    ds = ds.assign(mask_ground_return=mask_ground_return)
    ds["mask_ground_return"].attrs = {
        "long_name": "Mask for ground reflections",
        "description": "1 indicates no ground reflection, 0 indicates ground reflection",
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

    mask_roll = (np.abs(ds.plane_roll) <= 5).astype(int)
    mask_roll.attrs = {
        "long_name": "Mask for roll segments",
        "description": "0 indicates roll higher 5 deg, 1 indicates roll lower 5 deg",
    }
    ds = ds.assign(mask_roll=mask_roll)

    return ds


def add_georeference(ds, lat, lon, plane_pitch, plane_roll, plane_altitude, source):
    """Add georeference information to dataset."""
    ds = ds.assign(
        plane_altitude=plane_altitude.sel(time=ds.time, method="nearest").assign_coords(
            time=ds.time
        ),
        lat=lat.sel(time=ds.time, method="nearest").assign_coords(time=ds.time),
        lon=lon.sel(time=ds.time, method="nearest").assign_coords(time=ds.time),
        plane_roll=plane_roll.sel(time=ds.time, method="nearest").assign_coords(
            time=ds.time
        ),
        plane_pitch=plane_pitch.sel(time=ds.time, method="nearest").assign_coords(
            time=ds.time
        ),
    )
    ds.attrs["georeference source"] = source
    return ds


def format_bahamas(ds):
    """Post-processing of BAHAMAS datasets."""
    return ds.pipe(
        _bahamas_fix_time,
    )


def format_radiometer(ds):
    """Post-processing of radiometer datasets."""
    return (
        ds.rename(number_frequencies="frequency")
        .set_index(frequency="Freq")
        .pipe(
            _fix_radiometer_time,
        )
    )


def format_iwv(ds):
    """Post-processing of IWV datasets."""
    return ds.pipe(_fix_radiometer_time)


def format_radar(ds):
    """Post-processing of Radar quick look datasets."""
    return ds.pipe(
        _radar_fix_time,
    ).pipe(
        _radar_add_dBZ,
    )


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


def filter_radiometer(ds):
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

    return ds.pipe(_altitude_filter).pipe(_trim_dataset)


def add_masks_radiometer(ds, sea_land_mask):
    """Add masks to radiometer data for sea and land.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset.
    sea_land_mask : xr.DataArray
        Mask of land and sea. 1 for sea, 0 for land.

    Returns
    -------
    xr.Dataset
        Radiometer dataset with masks for height and roll angle.
    """

    return ds.pipe(_add_sea_land_mask, sea_land_mask).pipe(_add_amplifier_fault_mask)


def add_masks_iwv(ds, sea_land_mask):
    """Add masks to IWV data for sea and land.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 IWV dataset.
    sea_land_mask : xr.DataArray
        Mask of land and sea. 1 for sea, 0 for land.

    Returns
    -------
    xr.Dataset
        IWV dataset with masks for height and roll angle.
    """

    return ds.pipe(_add_sea_land_mask, sea_land_mask)


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

    ds.attrs["flight_id"] = flight_id
    ds.attrs["title"] = (
        "MIRA Cloud Radar Moments from the HALO Microwave Package (Level 3)"
    )
    ds.attrs["summary"] = (
        "This dataset contains measurements from the MIRA cloud radar onboard the HALO aircraft during the ORCESTRA campaign."
        "The measurements are processed and quality controlled. The processing includes formatting the data, adding georeference information,"
        "filtering for noise, clutter and valid radar states, and adding masks for calibration, ground reflections, and roll segments."
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
    ds.attrs["license"] = "CC-BY-4.0"
    ds.attrs["featureType"] = "trajectoryProfile"
    ds.attrs["references"] = "10.5194/amt-12-1815-2019, 10.5194/essd-13-5545-2021"
    ds.attrs["keywords"] = "Cloud Radar, HALO, ORCESTRA, PERCUSION, Tropical Atlantic"

    ds.attrs.pop("Copywright", None)
    ds.attrs.pop("Copywright_Owner", None)
    ds.attrs.pop("Latitude", None)
    ds.attrs.pop("Longitude", None)
    ds.attrs.pop("Altitude", None)
    ds.attrs.pop("institution", None)
    ds.attrs.pop("location", None)

    return ds


def add_metadata_radiometer(ds, flight_id):
    """
    Add metadata to the radiometer dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset.
    Returns
    -------
    xr.Dataset
        Radiometer dataset with metadata.
    """

    ds.attrs["flight_id"] = flight_id
    ds.attrs["title"] = "Radiometer Data from the Halo Microwave Package (Level 2)"
    ds.attrs["summary"] = (
        "This dataset contains measurements from the radiometers onboard the HALO aircraft during the ORCESTRA campaign."
        "The measurements are processed and quality controlled. The processing includes formatting the data,"
        "adding georeference information, and filtering for altitudes above 4800 m and adding a land-sea mask."
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
    ds.attrs["license"] = "CC-BY-4.0"
    ds.attrs["featureType"] = "trajectoryProfile"
    ds.attrs["references"] = "10.5194/amt-12-1815-2019, 10.5194/essd-13-5545-2021"
    ds.attrs["keywords"] = (
        "Radiometer, Microwave, HALO, ORCESTRA, PERCUSION, Tropical Atlantic"
    )

    ds.attrs.pop("Comment", None)
    ds.attrs.pop("Radiometer_Location", None)
    ds.attrs.pop("Station_Altitude", None)
    ds.attrs.pop("Station_Latitude", None)
    ds.attrs.pop("Station_Longitude", None)
    ds.attrs.pop("Radiometer_Software", None)
    ds.attrs.pop("Serial_Number", None)

    return ds


def add_metadata_iwv(ds, flight_id):
    """
    Add metadata to the IWV dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 IWV dataset.
    Returns
    -------
    xr.Dataset
        IWV dataset with metadata.
    """

    ds.attrs["flight_id"] = flight_id
    ds.attrs["title"] = (
        "Integrated Water Vapor Retrieval from the K-Band and W-Band Microwave Radiometers"
    )
    ds.attrs["summary"] = (
        "This dataset contains retrievals of integrated water vapor (IWV) from the K-band and W-band microwave radiometers onboard the HALO aircraft during the ORCESTRA campaign."
        "The retrieval is based on a linear regression model and should not be used for quantitative analysis."
        "The purpose of this data was to provide an estimate of IWV during the flights. The processing includes formatting the data,"
        "adding georeference information, filtering for altitudes above 4800 m and adding a land-sea mask."
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
    ds.attrs["license"] = "CC-BY-4.0"
    ds.attrs["featureType"] = "trajectoryProfile"
    ds.attrs["references"] = "10.5194/amt-7-4539-2014"
    ds.attrs["keywords"] = (
        "Radiometer, Microwave, Integrated Water Vapor, Retrieval, HALO, ORCESTRA, PERCUSION, Tropical Atlantic"
    )

    ds.attrs.pop("Comment", None)
    ds.attrs.pop("Radiometer_Location", None)
    ds.attrs.pop("Station_Altitude", None)
    ds.attrs.pop("Station_Latitude", None)
    ds.attrs.pop("Station_Longitude", None)
    ds.attrs.pop("Radiometer_Software", None)
    ds.attrs.pop("Serial_Number", None)
    ds.attrs.pop("Host-PC_Software", None)
    ds.attrs.pop("Radiometer_System", None)

    return ds
