# %%
from src.plots_functions import testplot_hamp
import xarray as xr
import yaml
import pandas as pd

# %% load data
flights = pd.read_csv("flights.csv", index_col=0)

with open("process_config.yaml", "r") as file:
    cfg = yaml.safe_load(file)

radar = xr.open_dataset(f"{cfg['save_dir']}/full_radar.zarr", engine="zarr")
radiometers = xr.open_dataset(
    f"{cfg['save_dir']}/full_radiometer.zarr",
    engine="zarr",
)
iwv = xr.open_dataset(f"{cfg['save_dir']}/full_iwv.zarr", engine="zarr")

# %%
for date in [20240929]:
    if date == 20240929:  # only flight that crossed 0 UTC
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
        )
    fig.savefig(f"Plots/{date}_hamp.png", dpi=300)

# %%
