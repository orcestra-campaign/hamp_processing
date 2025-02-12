# %%
import src.readwrite_functions as rwfuncs
from src.post_processed_hamp_data import PostProcessedHAMPData
from src.plots_functions import testplot_hamp
import xarray as xr

# %% load data
cfg = rwfuncs.extract_config_params("config_ipns.yaml")

hampdata = PostProcessedHAMPData(
    xr.open_dataset(cfg["path_radar"], engine="zarr"),
    xr.open_dataset(cfg["path_radiometers"], engine="zarr"),
    xr.open_dataset(cfg["path_iwv"], engine="zarr"),
)

# %% plot
testplot_hamp(hampdata.radar, hampdata.radiometers)

# %%
