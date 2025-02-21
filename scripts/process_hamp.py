# %% import modules
import os
import yaml
import xarray as xr
import shutil
import pandas as pd
from src.process import (
    format_radiometer,
    format_radar,
    format_iwv,
    filter_radar,
    filter_radiometer,
    add_masks_radar,
    add_masks_radiometer,
    add_masks_iwv,
    add_metadata_radar,
    add_metadata_radiometer,
    add_metadata_iwv,
    add_georeference,
    correct_radar_height,
    cleanup_iwv,
    cleanup_radiometers,
)
from src.ipfs_helpers import get_encoding, read_nc, read_mf_nc


# %% define function


def postprocess_hamp(date, flightletter, version):
    """
    Postprocess raw data from HAMP.

    Parameters
    ----------
    date : str
        Date of the data in the format YYYYMMDD
    flightletter : str
        Letter of the flight
    version : str
        Version number of the processed data

    Returns
    -------
    None
    """

    # load config file
    config_file = "process_config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # configure paths
    paths = {}
    paths["radar"] = config["radar"].format(date=date, flightletter=flightletter)
    # dirty fix
    if date == "20240809":
        paths["radiometer"] = config["radiometer"].format(date=date, flightletter="a")
    else:
        paths["radiometer"] = config["radiometer"].format(
            date=date, flightletter=flightletter
        )
    paths["bahamas"] = config["bahamas"].format(date=date, flightletter=flightletter)
    paths["sea_land_mask"] = config["sea_land_mask"]
    paths["save_dir"] = config["save_dir"].format(date=date, flightletter=flightletter)

    # load raw data
    print(f"Loading raw data for {date}")
    ds_radar_raw = read_mf_nc(paths["radar"]).load()
    ds_bahamas = (
        xr.open_dataset(paths["bahamas"], engine="zarr")
        .reset_coords(["lat", "lon", "alt"])
        .resample(time="0.25s")
        .mean()
    )
    if date == "20240929":  # only flight that crossed 0 UTC
        date_2 = "20240930"
        ds_iwv_raw_29 = read_nc(f"{paths['radiometer']}/KV/{date[2:]}.IWV.NC")
        ds_iwv_raw_30 = read_nc(f"{paths['radiometer']}/KV/{date_2[2:]}.IWV.NC")
        ds_iwv_raw = xr.combine_by_coords(
            [ds_iwv_raw_29, ds_iwv_raw_30],
            data_vars="minimal",
            compat="override",
            coords="minimal",
        )
        radiometers = ["183", "11990", "KV"]
        ds_radiometers_raw_29 = {}
        ds_radiometers_raw_30 = {}
        ds_radiometers_raw = {}
        for radio in radiometers:
            ds_radiometers_raw_29[radio] = read_nc(
                f"{paths['radiometer']}/{radio}/{date[2:]}.BRT.NC"
            )
            ds_radiometers_raw_30[radio] = read_nc(
                f"{paths['radiometer']}/{radio}/{date_2[2:]}.BRT.NC"
            )
            ds_radiometers_raw[radio] = xr.combine_by_coords(
                [ds_radiometers_raw_29[radio], ds_radiometers_raw_30[radio]],
                data_vars="minimal",
                compat="override",
                coords="minimal",
            )

    else:
        ds_iwv_raw = read_nc(f"{paths['radiometer']}/KV/{date[2:]}.IWV.NC")
        radiometers = ["183", "11990", "KV"]
        ds_radiometers_raw = {}
        for radio in radiometers:
            ds_radiometers_raw[radio] = read_nc(
                f"{paths['radiometer']}/{radio}/{date[2:]}.BRT.NC"
            )

    print("Processing")
    ds_radar_lev1 = format_radar(ds_radar_raw).pipe(
        add_georeference,
        lat=ds_bahamas["lat"],
        lon=ds_bahamas["lon"],
        plane_pitch=ds_bahamas["pitch"],
        plane_roll=ds_bahamas["roll"],
        plane_altitude=ds_bahamas["alt"],
        source=ds_bahamas.attrs["source"],
    )
    ds_iwv_lev1 = format_iwv(ds_iwv_raw).pipe(
        add_georeference,
        lat=ds_bahamas["lat"],
        lon=ds_bahamas["lon"],
        plane_pitch=ds_bahamas["pitch"],
        plane_roll=ds_bahamas["roll"],
        plane_altitude=ds_bahamas["alt"],
        source=ds_bahamas.attrs["source"],
    )
    ds_radiometers_lev1 = {}
    for radio in radiometers:
        ds_radiometers_lev1[radio] = format_radiometer(ds_radiometers_raw[radio])

    # concatenate radiometers and add georeference
    ds_radiometers_lev1_concat = xr.concat(
        [ds_radiometers_lev1[radio] for radio in radiometers], dim="frequency"
    ).sortby("frequency")
    ds_radiometers_lev1_concat = ds_radiometers_lev1_concat.assign(
        TBs=ds_radiometers_lev1_concat["TBs"].T
    ).pipe(
        add_georeference,
        lat=ds_bahamas["lat"],
        lon=ds_bahamas["lon"],
        plane_pitch=ds_bahamas["pitch"],
        plane_roll=ds_bahamas["roll"],
        plane_altitude=ds_bahamas["alt"],
        source=ds_bahamas.attrs["source"],
    )

    sea_land_mask = xr.open_dataarray(paths["sea_land_mask"]).load()

    ds_radar_lev2 = (
        ds_radar_lev1.pipe(correct_radar_height)
        .pipe(filter_radar)
        .pipe(add_metadata_radar, flight_id=f"{date}{flightletter}")
        .pipe(add_masks_radar, sea_land_mask)
        .transpose("time", "height")
    )
    ds_radiometer_lev2 = (
        filter_radiometer(ds_radiometers_lev1_concat)
        .pipe(add_masks_radiometer, sea_land_mask)
        .pipe(add_metadata_radiometer, flight_id=f"{date}{flightletter}")
        .pipe(cleanup_radiometers)
        .transpose("time", "frequency")
    )
    ds_iwv_lev2 = (
        filter_radiometer(ds_iwv_lev1)
        .pipe(add_metadata_iwv, flight_id=f"{date}{flightletter}")
        .pipe(add_masks_iwv, sea_land_mask)
        .pipe(cleanup_iwv)
    )

    print(f"Saving data for {date}")
    ds_radar_lev2.attrs["version"] = version
    ds_radiometer_lev2.attrs["version"] = version
    ds_iwv_lev2.attrs["version"] = version
    # radar
    path_radar = f"{paths['save_dir']}/radar/HALO-{date}{flightletter}_radar.zarr"
    if os.path.exists(path_radar):
        shutil.rmtree(path_radar)
    ds_radar_lev2.chunk(time=4**9, height=-1).to_zarr(
        path_radar, encoding=get_encoding(ds_radar_lev2), mode="w"
    )
    # radiometer
    path_radiometer = (
        f"{paths['save_dir']}/radiometer/HALO-{date}{flightletter}_radio.zarr"
    )
    if os.path.exists(path_radiometer):
        shutil.rmtree(path_radiometer)
    ds_radiometer_lev2.chunk(time=4**9, frequency=-1).to_zarr(
        path_radiometer, encoding=get_encoding(ds_radiometer_lev2), mode="w"
    )
    # iwv
    path_iwv = f"{paths['save_dir']}/iwv/HALO-{date}{flightletter}_iwv.zarr"
    if os.path.exists(path_iwv):
        shutil.rmtree(path_iwv)
    ds_iwv_lev2.chunk(time=4**9).to_zarr(
        path_iwv, encoding=get_encoding(ds_iwv_lev2), mode="w"
    )


# %% run postprocessing
flights = pd.read_csv("flights.csv", index_col=0)
version = "1.0"
for date, flightletter in zip(flights.index, flights["flightletter"]):
    postprocess_hamp(str(date), flightletter, version)


# %%
