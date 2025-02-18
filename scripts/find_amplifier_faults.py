# %%
import matplotlib.pyplot as plt
import xarray as xr
import yaml
from datetime import datetime

# %% create HAMP post-processed data
with open("process_config.yaml", "r") as file:
    cfg = yaml.safe_load(file)

radar = xr.open_dataset(f"{cfg['save_dir']}/full_radar.zarr", engine="zarr")
radiometers = xr.open_dataset(
    f"{cfg['save_dir']}/full_radiometer.zarr",
    engine="zarr",
)
iwv = xr.open_dataset(f"{cfg['save_dir']}/full_iwv.zarr", engine="zarr")

# %% load raw radiometer data
ds_rad_raw = {}
for radio in ["11990", "KV", "183"]:
    ds_rad_raw[radio] = xr.open_dataset(
        f"Data/Radiometer_Data/HALO-{cfg['date']}a/{radio}/{cfg['date'][2:]}.LV0.NC"
    )

# %% plot radiometer data


def define_module(radio):
    if (radio == "K") | (radio == "V"):
        module = "KV"
        is_90 = False
    elif radio == "183":
        module = "183"
        is_90 = False
    elif radio == "90":
        module = "11990"
        is_90 = True
    elif radio == "119":
        module = "11990"
        is_90 = False
    else:
        raise ValueError("Invalid radiometer frequency")
    return module, is_90


freqs = {
    "K": slice(22, 32),
    "V": slice(50, 58),
    "183": slice(183, 191),
    "119": slice(120, 128),
    "90": 90,
}
radio = "119"
module, is_90 = define_module(radio)
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col", width_ratios=[10, 0.2])
ds_rad_raw[module].sel(frequency=freqs[radio])["gain"].plot.line(
    ax=axes[2, 0], x="time"
)
axes[1, 1].remove()
axes[2, 1].remove()
plt.show()

# %% filter radiometers
ds = 1
date = cfg["date"]
# read error file
with open(f"Data/error_files/HALO-{date}a.yaml") as file:
    error_file = yaml.safe_load(file)

# filter data
date_format = "%H:%M:%S"
variables_2d = [
    var for var in ds.radiometers.data_vars if ds.radiometers[var].ndim == 2
]
for radio in ["K", "V", "90", "119", "183"]:
    if radio in error_file.keys():
        for timeframe in error_file[radio]["times"]:
            start_time = datetime.strptime(timeframe[0], date_format).time()
            end_time = datetime.strptime(timeframe[1], date_format).time()
            mask = (ds.radiometers.time.dt.time < start_time) | (
                ds.radiometers.time.dt.time > end_time
            )
            for var in variables_2d:
                value = ds.radiometers[var]
                for f in ds.radiometers.sel(frequency=freqs[radio]).frequency.values:
                    value.loc[{"frequency": f}] = value.loc[{"frequency": f}].where(
                        mask
                    )
                ds.radiometers.assign({var: value})

# %% control plot

plt.show()


# %%
