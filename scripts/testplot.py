# %%
from src.plots_functions import testplot_hamp
import xarray as xr
import yaml
import pandas as pd

# %% load config
flights = pd.read_csv("flights.csv", index_col=0)

with open("process_config.yaml", "r") as file:
    cfg = yaml.safe_load(file)

# %% load full data
radar = xr.open_dataset(f"{cfg['save_dir']}/full_radar.zarr", engine="zarr")
radiometers = xr.open_dataset(
    f"{cfg['save_dir']}/full_radiometer.zarr",
    engine="zarr",
)
iwv = xr.open_dataset(f"{cfg['save_dir']}/full_iwv.zarr", engine="zarr")

# %% plot all flights
for date in flights.index:
    if date == 20240928:  # only flight that crossed 0 UTC
        fig = testplot_hamp(
            radar.sel(time=slice("20240929", "20240930")),
            radiometers.sel(time=slice("20240929", "20240930")),
            iwv.sel(time=slice("20240929", "20240930")),
            ground_filter=True,
            roll_filter=True,
            calibration_filter=True,
        )
    else:
        fig = testplot_hamp(
            radar.sel(time=str(date)),
            radiometers.sel(time=str(date)),
            iwv.sel(time=str(date)),
            ground_filter=True,
            roll_filter=True,
            calibration_filter=True,
            amplifier_faults=True,
        )
    fig.savefig(f"Plots/{date}_hamp_filtered.png", dpi=300)

# %% plot single flight
date = "20240916"
flightletter = "a"
radar = xr.open_dataset(
    f"{cfg['save_dir']}/radar/HALO-{date}{flightletter}_radar.zarr", engine="zarr"
)
radiometers = xr.open_dataset(
    f"{cfg['save_dir']}/radiometer/HALO-{date}{flightletter}_radio.zarr", engine="zarr"
)
iwv = xr.open_dataset(
    f"{cfg['save_dir']}/iwv/HALO-{date}{flightletter}_iwv.zarr", engine="zarr"
)

# %%
fig = testplot_hamp(
    radar,
    radiometers,
    iwv,
    ground_filter=True,
    roll_filter=True,
    calibration_filter=True,
    amplifier_faults=True,
)

# %%
