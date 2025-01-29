# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from orcestra.postprocess.level1 import (
    _filter_clutter,
    _add_clibration_mask,
    _add_ground_mask,
    _add_roll_mask,
    _add_metadata,
)

# import matplotlib

# %%


# %% load data
flight_id = "HALO-20240926a"
ds_radar = xr.open_dataset(
    f"ipns://latest.orcestra-campaign.org/products/HALO/radar/moments/{flight_id}.zarr",
    engine="zarr",
).pipe(_add_metadata, flight_id)

sea_land_mask = xr.open_dataset("/work/bm1183/m301049/orcestra/sea_land_mask.nc")

# %%
ds_radar = _add_clibration_mask(ds_radar)
ds_radar = _add_ground_mask(ds_radar, sea_land_mask)
ds_radar = _add_roll_mask(ds_radar)
ds_radar = _add_metadata(ds_radar)

# %% filter ground signal based on reflection maxima


# %%
ds_radar = _add_ground_mask(ds_radar, sea_land_mask)

# %%
plt.close("all")
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
pcol = axes[0].pcolormesh(
    ds_radar.time,
    ds_radar.height,
    ds_radar.dBZg.T,
    cmap="YlGnBu",
    vmin=-30,
    vmax=40,
)


axes[1].pcolormesh(
    ds_radar.time,
    ds_radar.height,
    ds_radar.dBZg.where(~ds_radar["mask_ground_return"]).T,
    cmap="YlGnBu",
    vmin=-30,
    vmax=40,
)


plt.show()

# %% try clutter filter on radar data

radar_data = ds_radar.sel(time=slice("2024-09-26 14:30", "2024-09-26 14:35"))
filtered_data = _filter_clutter(radar_data)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

radar_binary = ~np.isnan(radar_data["dBZg"])
filtered_binary = ~np.isnan(filtered_data["dBZg"])

radar_binary.plot.pcolormesh(ax=axes[0], cmap="gray", x="time", add_colorbar=False)
axes[0].set_title("Original Data Binary")

filtered_binary.plot.pcolormesh(ax=axes[1], cmap="gray", x="time", add_colorbar=False)
axes[1].set_title("Filtered Data Binary")


# %% loop over all 2D vars and check if they contain clutter


# %% try ground filter based on etopo data
ds_topo = xr.open_dataset("/work/bm1183/m301049/orcestra/etopo.nc")

# %%
ds_topo["z"].sel(lat=slice(-30, 30)).plot.contourf()


# %%
