# %%
import os
import xarray as xr
import yaml
import pandas as pd
import typhon
import pyarts
from tqdm import tqdm
import FluxSimulator as fsm
import numpy as np
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

# from src.plots_functions import plot_dropsonde, plot_TB_comparison


# %% get ARTS data
pyarts.cat.download.retrieve(verbose=True)

# %% setup sensor
sensor_description, NeDT, Accuracy, FWHM_Antenna = Hamp_channels(
    ["K", "V", "W", "F", "G"], rel_mandatory_grid_spacing=1.0 / 60.0
)
freqs = sensor_description[:, 0] + sensor_description[:, 1] + sensor_description[:, 2]

# %% setup workspace
ws = basic_setup([], sensor_description=sensor_description)

# %% load data
configfile = "process_config.yaml"
with open(configfile, "r") as file:
    cfg = yaml.safe_load(file)

ds_dropsonde = xr.open_dataset(
    "ipns://latest.orcestra-campaign.org/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    engine="zarr",
)
ds_dropsonde = ds_dropsonde.assign_coords(sonde=np.arange(ds_dropsonde.sonde.size))


# %% define function
def calc_arts_bts(date, flightletter, ds_dropsonde, cfg):
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

    flightname = f"HALO-{date}{flightletter}"
    ds_radar = xr.open_dataset(
        f"{cfg['save_dir']}/radar/{flightname}_radar.zarr", engine="zarr"
    )
    ds_radiometers = xr.open_dataset(
        f"{cfg['save_dir']}/radiometer/{flightname}_radio.zarr", engine="zarr"
    )
    ds_dropsonde = ds_dropsonde.where(
        (ds_dropsonde["interp_time"] > pd.to_datetime(date))
        & (ds_dropsonde["interp_time"] < pd.to_datetime(date) + pd.DateOffset(hour=23))
    ).dropna(dim="sonde", how="all")

    print("Calculate Cloud Mask")
    ds_dropsonde = ds_dropsonde.assign(
        radar_cloud_flag=(
            ds_radar.sel(time=ds_dropsonde.sonde_time, method="nearest").sel(
                height=slice(200, None)
            )["dBZe"]
            > -30
        ).max("height")
        * 1
    )
    cloud_free_idxs = (
        ds_dropsonde["sonde"]
        .where((ds_dropsonde["radar_cloud_flag"] == 0).compute(), drop=True)
        .values
    )

    freqs_hamp = ds_radiometers.frequency.values
    TBs_arts = pd.DataFrame(index=freqs / 1e9, columns=cloud_free_idxs)
    TBs_hamp = pd.DataFrame(index=freqs_hamp, columns=cloud_free_idxs)

    print("Setup Folders")
    if not os.path.exists(f"Data/arts_calibration/{flightname}"):
        os.makedirs(f"Data/arts_calibration/{flightname}")
    if not os.path.exists(f"Data/arts_calibration/{flightname}/plots"):
        os.makedirs(f"Data/arts_calibration/{flightname}/plots")

    print(f"Running {cloud_free_idxs.size} dropsondes for {date}")
    for sonde in tqdm(cloud_free_idxs):
        ds_dropsonde_loc, radiometers_loc, height, drop_time = get_profiles(
            sonde, ds_dropsonde, ds_radiometers
        )

        if not is_complete(ds_dropsonde_loc, radiometers_loc, drop_time, height, sonde):
            continue

        surface_temp = get_surface_temperature(ds_dropsonde_loc)
        surface_ws = get_surface_windspeed(ds_dropsonde_loc)

        ds_dropsonde_extrap = extrapolate_dropsonde(ds_dropsonde_loc, height)

        # plot_dropsonde(ds_dropsonde_extrap, ds_dropsonde_loc)

        profile_grd = fsm.generate_gridded_field_from_profiles(
            pyarts.arts.Vector(ds_dropsonde_extrap["p"].values),
            ds_dropsonde_extrap["ta"].values,
            ds_dropsonde_extrap["altitude"].values,
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

        TBs_arts[sonde] = pd.DataFrame(data=BTs, index=freqs / 1e9)
        TBs_hamp[sonde] = radiometers_loc["TBs"]

        # plot_TB_comparison(TBs_arts[sonde], TBs_hamp[sonde], sonde)

    TBs_arts.to_csv(f"Data/arts_calibration/{flightname}/TBs_arts.csv")
    TBs_hamp.to_csv(f"Data/arts_calibration/{flightname}/TBs_hamp.csv")


# %% loop over flights
flights = pd.read_csv("flights.csv", index_col=0)
flights_processed = flights[
    (flights["location"] == "sal") | (flights["location"] == "barbados")
]

for date in flights_processed.index:
    calc_arts_bts(
        str(date), flights_processed.loc[date]["flightletter"], ds_dropsonde, cfg
    )
# %%
