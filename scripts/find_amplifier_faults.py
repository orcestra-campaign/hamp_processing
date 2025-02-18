# %%
import xarray as xr
import yaml
from src.ipfs_helpers import read_nc
from src.plots_functions import plot_radiometers

# %% load processed data
with open("process_config.yaml", "r") as file:
    cfg = yaml.safe_load(file)

radar = xr.open_dataset(f"{cfg['save_dir']}/full_radar.zarr", engine="zarr")
radiometers = xr.open_dataset(
    f"{cfg['save_dir']}/full_radiometer.zarr",
    engine="zarr",
)
iwv = xr.open_dataset(f"{cfg['save_dir']}/full_iwv.zarr", engine="zarr")

# %% load raw data
date = "20241119"
flightletter = "a"
ds_rad_raw = {}
for radio in ["11990", "KV", "183"]:
    ds_rad_raw[radio] = read_nc(
        f"{cfg['radiometer'].format(date=date, flightletter=flightletter)}/{radio}/{date[2:]}.LV0.NC"
    )


# %% plot TBs and gain
plot_radiometers(
    radiometers.sel(time=date),
    ds_rad_raw,
)

# %%
