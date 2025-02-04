# %%
import os
import xarray as xr
from src import load_data_functions as loadfuncs
from src.arts_functions import (
    Hamp_channels,
    basic_setup,
    extrapolate_dropsonde,
    get_profiles,
    get_surface_temperature,
    get_surface_windspeed,
    forward_model,
)
from src.plot_functions import plot_arts_flux
from src.ipfs_helpers import read_nc
from orcestra.postprocess.level0 import bahamas
from src import readwrite_functions as rwfuncs
import yaml
import pandas as pd
import typhon
import pyarts
from tqdm import tqdm
import FluxSimulator as fsm


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
def calc_arts_bts(date):
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
    # change the date in config.yaml to date
    configfile = "config_ipns.yaml"
    with open(configfile, "r") as file:
        config_yml = yaml.safe_load(file)
    config_yml["date"] = date
    with open(configfile, "w") as file:
        yaml.dump(config_yml, file)
    # read config
    cfg = rwfuncs.extract_config_params(configfile)

    # load bahamas data from ipfs
    print("Load Bahamas Data")
    ds_bahamas = (
        read_nc(
            f"ipns://latest.orcestra-campaign.org/raw/HALO/bahamas/{cfg['flightname']}/QL_*.nc"
        )
        .pipe(bahamas)
        .interpolate_na("time")
    )

    # read dropsonde data
    print("Load Dropsonde Data")
    ds_dropsonde = xr.open_dataset(cfg["path_dropsondes"], engine="zarr")
    ds_dropsonde = ds_dropsonde.where(
        (ds_dropsonde["interp_time"] > pd.to_datetime(cfg["date"]))
        & (
            ds_dropsonde["interp_time"]
            < pd.to_datetime(cfg["date"]) + pd.DateOffset(hour=23)
        )
    ).dropna(dim="sonde_id", how="all")

    # read HAMP post-processed data
    print("Load HAMP Data")
    hampdata = loadfuncs.load_hamp_data(
        cfg["path_radar"], cfg["path_radiometers"], cfg["path_iwv"]
    )

    # cloud mask from radar
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

    # initialize result arrays
    freqs_hamp = hampdata.radiometers.frequency.values
    TBs_arts = pd.DataFrame(index=freqs_hamp, columns=cloud_free_idxs)
    TBs_hamp = TBs_arts.copy()

    # setup folders
    print("Setup Folders")
    if not os.path.exists(f"Data/arts_calibration/{cfg['flightname']}"):
        os.makedirs(f"Data/arts_calibration/{cfg['flightname']}")
    if not os.path.exists(f"Data/arts_calibration/{cfg['flightname']}/plots"):
        os.makedirs(f"Data/arts_calibration/{cfg['flightname']}/plots")

    # loop over cloud free sondes
    print(f"Running {cloud_free_idxs.size} dropsondes for {cfg['flightname']}")
    for sonde_id in tqdm(cloud_free_idxs):
        # get profiles
        ds_dropsonde_loc, hampdata_loc, height, drop_time = get_profiles(
            sonde_id, ds_dropsonde, hampdata
        )

        # check if dropsonde is broken (contains only nan values)
        if ds_dropsonde_loc["ta"].isnull().mean().values == 1:
            print(f"Dropsonde {sonde_id} is broken, skipping")
            continue

        # get surface values
        surface_temp = get_surface_temperature(ds_dropsonde_loc)
        surface_ws = get_surface_windspeed(ds_dropsonde_loc)

        # extrapolate dropsonde profiles
        ds_dropsonde_extrap = extrapolate_dropsonde(
            ds_dropsonde_loc, height, ds_bahamas
        )

        # convert xarray to ARTS gridded field
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

        # run arts
        BTs, _ = forward_model(
            ws,
            profile_grd,
            surface_ws,
            surface_temp,
            height,
        )

        # except (ValueError, KeyError, RuntimeError) as e:
        #    print(
        #        f"ARTS or extrapolation failed for dropsonde {sonde_id} with error: {e}, skipping"
        #    )
        #    pass

        # Store arts BTs
        TBs_arts[sonde_id] = pd.DataFrame(data=BTs, index=freqs / 1e9)
        # get according hamp data
        TBs_hamp[sonde_id] = hampdata_loc.radiometers.TBs.values

        # Plot to compare arts to hamp radiometers
        fig, ax = plot_arts_flux(
            TBs_hamp[sonde_id],
            TBs_arts[sonde_id],
            dropsonde_id=sonde_id,
            time=drop_time,
            ds_bahamas=ds_bahamas,
        )
        fig.savefig(f"Data/arts_calibration/{cfg['flightname']}/plots/{sonde_id}.png")
        fig.clf()

    # save results
    TBs_arts.to_csv(f"Data/arts_calibration/{cfg['flightname']}/TBs_arts.csv")
    TBs_hamp.to_csv(f"Data/arts_calibration/{cfg['flightname']}/TBs_hamp.csv")


# %% call function
# date = str(sys.argv[1])
calc_arts_bts("20240827")

# %% test turning extrapolated dropsonde profiles to ARTS input
# read config
cfg = rwfuncs.extract_config_params("config_ipns.yaml")
ds_dropsonde = xr.open_dataset(cfg["path_dropsondes"], engine="zarr")
profile = ds_dropsonde.isel(sonde_id=100)
profile = profile.dropna("gpsalt")

# %% convert xarray to ARTS gridded field
profile_grd = fsm.generate_gridded_field_from_profiles(
    pyarts.arts.Vector(profile["p"].values),
    profile["ta"].values,
    profile["alt"].values,
    gases={
        "H2O": typhon.physics.specific_humidity2vmr(profile["q"].values),
        "N2": 0.78,
        "O2": 0.21,
    },
)

# %%

BTs, _ = forward_model(
    ws,
    profile_grd,
    10,
    300,
    1e4,
)
# %%
freqs = sensor_description[:, 0] + sensor_description[:, 1] + sensor_description[:, 2]

# %%
