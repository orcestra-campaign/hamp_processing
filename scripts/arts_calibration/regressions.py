# %%
import numpy as np
from scipy.stats import linregress
import pandas as pd
import xarray as xr
from src import readwrite_functions as rwfuncs
from src.plots_functions import plot_regression

# %% Read csv BTs

flights = pd.read_csv("flights.csv", index_col=0)
flights_processed = flights[
    (flights["location"] == "sal") | (flights["location"] == "barbados")
]

TB_arts_list = []
TB_hamp_list = []
for date in flights_processed.index:
    TB_arts_list.append(
        pd.read_csv(f"Data/arts_calibration/HALO-{date}a/TBs_arts.csv", index_col=0)
    )
    TB_hamp_list.append(
        pd.read_csv(f"Data/arts_calibration/HALO-{date}a/TBs_hamp.csv", index_col=0)
    )

# load dropsonde data
configfile = "config_ipns.yaml"
cfg = rwfuncs.extract_config_params(configfile)
ds_dropsonde = xr.open_dataset(cfg["path_dropsondes"], engine="zarr")

# restructure data
TB_arts = pd.concat(TB_arts_list, axis=1)
TB_hamp = pd.concat(TB_hamp_list, axis=1)
launch_time = ds_dropsonde.sel(sonde_id=TB_arts.columns).launch_time.values
TB_arts.columns = launch_time
TB_hamp.columns = launch_time
TB_arts = TB_arts.T.dropna()
TB_hamp = TB_hamp.T.dropna()

# %% drop extreme outliers
TB_arts_std = TB_arts.std("index")
TB_hamp_std = TB_hamp.std("index")
TB_arts_med = TB_arts.median("index")
TB_hamp_med = TB_hamp.median("index")
TB_arts = TB_arts.where((TB_arts - TB_arts_med).abs() < 3 * TB_arts_std)
TB_hamp = TB_hamp.where((TB_hamp - TB_hamp_med).abs() < 3 * TB_hamp_std)

# %% count NaN values
mask_arts = ((TB_arts - TB_arts_med).abs() < 2 * TB_arts_std).mean("index")
mask_hamp = ((TB_hamp - TB_hamp_med).abs() < 2 * TB_hamp_std).mean("index")
mask_arts

# %%
mask_hamp


# %%
multi_index = pd.MultiIndex.from_product(
    [TB_arts.columns, ["slope", "intercept"]], names=["frequency", "parameter"]
)
regression_coeffs = pd.DataFrame(
    index=np.unique(TB_arts.index.date), columns=multi_index
)

for date in flights_processed.index:
    date = pd.to_datetime(date, format="%Y%m%d").date()
    TB_arts_day = TB_arts.loc[TB_arts.index.date == date]
    TB_hamp_day = TB_hamp.loc[TB_hamp.index.date == date]

    for f in TB_arts_day.columns:
        # Flatten the arrays and filter out NaN values
        x = TB_hamp_day[f].values.flatten()
        y = TB_arts_day[f].values.flatten()
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) > 0 and len(y) > 0:  # Ensure there are values left after filtering
            regression = linregress(x, y)
            regression_coeffs.loc[date, (f, "slope")] = regression.slope
            regression_coeffs.loc[date, (f, "intercept")] = regression.intercept

# %% plot regressions
plot_regression(regression_coeffs, TB_arts, TB_hamp, date)

# %%
