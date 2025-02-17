# %%
from src.plots_functions import testplot_hamp
import xarray as xr
import yaml
import pandas as pd

# %% load data
flights = pd.read_csv("flights.csv", index_col=0)

for date, flightletter in zip(flights.index, flights["flightletter"]):
    with open("process_config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    radar = xr.open_dataset(
        f"{cfg['save_dir']}/radar/HALO-{date}{flightletter}_radar.zarr", engine="zarr"
    )
    radiometers = xr.open_dataset(
        f"{cfg['save_dir']}/radiometer/HALO-{date}{flightletter}_radio.zarr",
        engine="zarr",
    )
    iwv = xr.open_dataset(
        f"{cfg['save_dir']}/iwv/HALO-{date}{flightletter}_iwv.zarr", engine="zarr"
    )

    # plot
    fig = testplot_hamp(
        radar,
        radiometers,
        iwv,
        ground_filter=True,
        roll_filter=True,
        calibration_filter=True,
    )
    fig.savefig(f"Plots/{date}{flightletter}_hamp.png", dpi=300)

# %%

ds = xr.open_dataset(
    "/work/bm1183/m301049/orcestra/Hamp_Processed/radar/HALO-20240809b_radar.zarr",
    engine="zarr",
)

# %%
