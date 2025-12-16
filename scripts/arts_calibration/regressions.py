# %%
import numpy as np
from scipy.stats import linregress
import pandas as pd
import xarray as xr
import yaml
from src.plots_functions import plot_regression
import matplotlib.pyplot as plt

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
with open("process_config.yaml") as f:
    cfg = yaml.safe_load(f)

ds_dropsonde = xr.open_dataset(
    "ipns://latest.orcestra-campaign.org/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    engine="zarr",
)
ds_dropsonde = ds_dropsonde.assign_coords(sonde=np.arange(ds_dropsonde.sonde.size))

# restructure data
TB_arts = pd.concat(TB_arts_list, axis=1)
TB_hamp = pd.concat(TB_hamp_list, axis=1)
launch_time = ds_dropsonde.sel(
    sonde=[int(float(x)) for x in TB_arts.columns.values]
).sonde_time.values
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
    [TB_arts.columns, ["slope", "standard_error"]], names=["frequency", "parameter"]
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
fig = plot_regression(regression_coeffs, TB_arts, TB_hamp, date)
fig.savefig(f"Plots/arts/{date}_regression.png", dpi=300)

# %% define frequencies
freq_k = [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]
freq_v = [50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]
freq_90 = [90.0]
center_freq_119 = 118.75
center_freq_183 = 183.31
width_119 = [1.4, 2.3, 4.2, 8.5]
width_183 = [0.6, 1.5, 2.5, 3.5, 5.0, 7.5]
freq_119 = [center_freq_119 + w for w in width_119]
freq_183 = [center_freq_183 + w for w in width_183]

calib_dates_183 = [
    "20240822",
    "20240901",
    "20240905",
    "20240913",
    "20240916",
    "20240918",
    "20240920",
    "20240923",
    "20240925",
    "20240928",
]

calib_oters = [
    "20240901",
    "20240905",
    "20240913",
    "20240916",
    "20240918",
    "20240920",
    "20240923",
    "20240925",
    "20240928",
]


def get_next_flight(calibration_date, flights):
    for flight in flights:
        if flight >= calibration_date.date():
            return flight
    return None


# %% plot slope and standard error for all modules
fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=False)

for i, freq in enumerate([freq_k, freq_v, freq_90, freq_119, freq_183]):
    ax = axes.flatten()[i]
    ax.axhline(1, color="grey", linewidth=0.5)
    for f in freq:
        ax.plot(
            regression_coeffs.index, regression_coeffs[f]["slope"], label=f"{f} GHz"
        )

    ax.set_ylabel("Slope")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    # format x-axis to show month and day
    ax.set_xticks(regression_coeffs.index)
    ax.set_xticklabels(regression_coeffs.index, rotation=45)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
    ax.set_xlim([regression_coeffs.index[0], regression_coeffs.index[-1]])


axes[-1, -1].remove()

fig.tight_layout()
fig.savefig("Data/arts_calibration/slope.png", dpi=300, bbox_inches="tight")


# %%
