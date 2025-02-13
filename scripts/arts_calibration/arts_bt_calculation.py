# %%
import os
import xarray as xr
import src.readwrite_functions as rwfuncs
from src.post_processed_hamp_data import PostProcessedHAMPData
import yaml
import pandas as pd
import typhon
import pyarts
from tqdm import tqdm
import FluxSimulator as fsm
from src.arts_functions import (
    Hamp_channels,
    basic_setup,
    extrapolate_dropsonde,
    get_profiles,
    get_surface_temperature,
    get_surface_windspeed,
    forward_model,
    is_complete,
)


# %% get ARTS data
print("Download ARTS data")
pyarts.cat.download.retrieve(verbose=True)

# %% setup sensor
sensor_description, NeDT, Accuracy, FWHM_Antenna = Hamp_channels(
    ["K", "V", "W", "F", "G"], rel_mandatory_grid_spacing=1.0 / 60.0
)
freqs = sensor_description[:, 0] + sensor_description[:, 1] + sensor_description[:, 2]

# %% setup workspace
ws = basic_setup([], sensor_description=sensor_description)


# %% define function
def calc_arts_bts(date, flightletter):
    """
    Calculates brightness temperatures for the radiometer frequencies with
    ARTS based on the dropsonde profiles for the flight on date.

    PARAMETERS
    ----------
    date: date on which flight took place (str)

    RETURN:
    ------
    None. Data is saved in arts_comparison folder.
    """

    print("Read Config")
    configfile = "config_ipns.yaml"
    with open(configfile, "r") as file:
        config_yml = yaml.safe_load(file)
    config_yml["date"] = date
    config_yml["flightletter"] = flightletter
    with open(configfile, "w") as file:
        yaml.dump(config_yml, file)
    cfg = rwfuncs.extract_config_params(configfile)

    print("Load Bahamas Data")
    ds_bahamas = xr.open_dataset(cfg["path_position_attitude"], engine="zarr")

    print("Load Dropsonde Data")
    ds_dropsonde = xr.open_dataset(cfg["path_dropsondes"], engine="zarr")
    ds_dropsonde = ds_dropsonde.where(
        (ds_dropsonde["interp_time"] > pd.to_datetime(cfg["date"]))
        & (
            ds_dropsonde["interp_time"]
            < pd.to_datetime(cfg["date"]) + pd.DateOffset(hour=23)
        )
    ).dropna(dim="sonde_id", how="all")

    print("Load HAMP Data")
    hampdata = PostProcessedHAMPData(
        xr.open_dataset(cfg["path_radar"], engine="zarr"),
        xr.open_dataset(cfg["path_radiometers"], engine="zarr"),
        xr.open_dataset(cfg["path_iwv"], engine="zarr"),
    )

    print("Calculate Cloud Mask")
    ds_dropsonde = ds_dropsonde.assign(
        radar_cloud_flag=(
            hampdata.radar.sel(time=ds_dropsonde.launch_time, method="nearest").sel(
                height=slice(200, None)
            )["dBZe"]
            > -30
        ).max("height")
        * 1
    )
    cloud_free_idxs = (
        ds_dropsonde["sonde_id"]
        .where(ds_dropsonde["radar_cloud_flag"] == 0, drop=True)
        .values
    )

    freqs_hamp = hampdata.radiometers.frequency.values
    TBs_arts = pd.DataFrame(index=freqs / 1e9, columns=cloud_free_idxs)
    TBs_hamp = pd.DataFrame(index=freqs_hamp, columns=cloud_free_idxs)

    print("Setup Folders")
    if not os.path.exists(f"Data/arts_calibration/{cfg['flightname']}"):
        os.makedirs(f"Data/arts_calibration/{cfg['flightname']}")
    if not os.path.exists(f"Data/arts_calibration/{cfg['flightname']}/plots"):
        os.makedirs(f"Data/arts_calibration/{cfg['flightname']}/plots")

    print(f"Running {cloud_free_idxs.size} dropsondes for {cfg['flightname']}")
    for sonde_id in tqdm(cloud_free_idxs):
        ds_dropsonde_loc, hampdata_loc, height, drop_time = get_profiles(
            sonde_id, ds_dropsonde, hampdata
        )

        if not is_complete(ds_dropsonde_loc, hampdata_loc, drop_time, height, sonde_id):
            continue

        surface_temp = get_surface_temperature(ds_dropsonde_loc)
        surface_ws = get_surface_windspeed(ds_dropsonde_loc)

        ds_dropsonde_extrap = extrapolate_dropsonde(
            ds_dropsonde_loc, height, ds_bahamas
        )

        # plot_dropsonde(ds_dropsonde_extrap, ds_dropsonde_loc)

        profile_grd = fsm.generate_gridded_field_from_profiles(
            pyarts.arts.Vector(ds_dropsonde_extrap["p"].values),
            ds_dropsonde_extrap["ta"].values,
            ds_dropsonde_extrap["alt"].values,
            gases={
                "H2O": typhon.physics.specific_humidity2vmr(
                    ds_dropsonde_extrap["q"].values
                ),
                "N2": 0.78,
                "O2": 0.21,
            },
        )

        BTs, _ = forward_model(
            ws,
            profile_grd,
            surface_ws,
            surface_temp,
            height,
        )

        TBs_arts[sonde_id] = pd.DataFrame(data=BTs, index=freqs / 1e9)
        TBs_hamp[sonde_id] = hampdata_loc["TBs"]

        # plot_TB_comparison(TBs_arts[sonde_id], TBs_hamp[sonde_id], sonde_id)

    TBs_arts.to_csv(f"Data/arts_calibration/{cfg['flightname']}/TBs_arts.csv")
    TBs_hamp.to_csv(f"Data/arts_calibration/{cfg['flightname']}/TBs_hamp.csv")


# %% loop over flights

flights = pd.read_csv("flights.csv", index_col=0)
flights_processed = flights[
    (flights["location"] == "sal") | (flights["location"] == "barbados")
]

for date in flights_processed.index:
    calc_arts_bts(str(date), flights_processed.loc[date]["flightletter"])


# %%
