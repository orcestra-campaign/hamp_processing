# %%
import xarray as xr
import yaml
from src.ipfs_helpers import add_encoding
from dask.diagnostics import ProgressBar

# %% open datasets as on file and drop conflicting attributes
with open("process_config.yaml", "r") as file:
    cfg = yaml.safe_load(file)

radar = xr.open_mfdataset(
    f"{cfg['save_dir']}/radar/*.zarr",
    engine="zarr",
    combine_attrs="drop_conflicts",
).chunk(
    {
        "time": 4**5,
        "height": 4**4,
    }
)

radiometer = xr.open_mfdataset(
    f"{cfg['save_dir']}/radiometer/*.zarr",
    engine="zarr",
    combine_attrs="drop_conflicts",
).chunk(
    {
        "time": 4**8,
        "frequency": 5,
    }
)
iwv = xr.open_mfdataset(
    f"{cfg['save_dir']}/iwv/*.zarr",
    engine="zarr",
    combine_attrs="drop_conflicts",
).chunk(
    {
        "time": 4**9,
    },
)

# %% save single files
print("Save radar")
with ProgressBar():
    radar.pipe(add_encoding).to_zarr(f"{cfg['save_dir']}/full_radar.zarr")

print("Save radiometer")
with ProgressBar():
    radiometer.pipe(add_encoding).to_zarr(f"{cfg['save_dir']}/full_radiometer.zarr")

print("Save iwv")
with ProgressBar():
    iwv.pipe(add_encoding).to_zarr(f"{cfg['save_dir']}/full_iwv.zarr")
