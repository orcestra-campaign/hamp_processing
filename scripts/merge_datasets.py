# %%
import xarray as xr
import yaml
from src.ipfs_helpers import get_encoding
from dask.diagnostics import ProgressBar

# %% open datasets as on file and drop conflicting attributes
with open("process_config.yaml", "r") as file:
    cfg = yaml.safe_load(file)

radar = (
    xr.open_mfdataset(
        f"{cfg['save_dir']}/radar/*.zarr",
        engine="zarr",
        combine="nested",
        concat_dim="time",
        combine_attrs="drop_conflicts",
        chunks={"time": -1, "height": -1},
    )
    .sortby("time")
    .chunk(
        {
            "time": 2**18,
            "height": -1,
        }
    )
).transpose("time", "height")
radiometer = (
    xr.open_mfdataset(
        f"{cfg['save_dir']}/radiometer/*.zarr",
        engine="zarr",
        combine="nested",
        concat_dim="time",
        combine_attrs="drop_conflicts",
        chunks={"time": -1, "frequency": -1},
    )
    .chunk(
        {
            "time": 2**18,
            "frequency": -1,
        }
    )
    .transpose("time", "frequency")
)

iwv = xr.open_mfdataset(
    f"{cfg['save_dir']}/iwv/*.zarr",
    engine="zarr",
    combine_attrs="drop_conflicts",
    combine="nested",
    concat_dim="time",
    chunks={"time": -1},
).chunk(
    {
        "time": 2**18,
    },
)

# %% save single files
print("Save radar")
with ProgressBar():
    radar.to_zarr(
        f"{cfg['save_dir']}/full_radar.zarr",
        encoding=get_encoding(radar),
        mode="w",
    )

print("Save radiometer")
with ProgressBar():
    radiometer.to_zarr(
        f"{cfg['save_dir']}/full_radiometer.zarr",
        encoding=get_encoding(radiometer),
        mode="w",
    )

print("Save iwv")
with ProgressBar():
    iwv.to_zarr(
        f"{cfg['save_dir']}/full_iwv.zarr",
        encoding=get_encoding(iwv),
        mode="w",
    )

# %%
