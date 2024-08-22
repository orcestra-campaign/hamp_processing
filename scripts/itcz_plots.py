# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from src import readwrite_functions as rwfuncs
import yaml
import matplotlib.pyplot as plt
from src import plot_functions as plotfuncs
from src import itcz_functions as itczfuncs
from src.plot_quicklooks import save_figure

# %%
### -------- USER PARAMETERS YOU MUST SET IN CONFIG.YAML -------- ###
configyaml = sys.argv[1]
with open(configyaml, "r") as file:
    print(f"Reading config YAML: '{configyaml}'")
    cfg = yaml.safe_load(file)
path_hampdata = cfg["paths"]["hampdata"]
hampdata = rwfuncs.load_timeslice_all_level1hampdata(path_hampdata, cfg["is_planet"])
### ------------------------------------------------------------- ###


# %% fuunctions for plotting HAMP post-processed data slice
def plot_radar_cwv_timeseries(
    hampdata,
    figsize=(9, 5),
    savefigparams=[],
):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=figsize, width_ratios=[18, 7], sharey="row"
    )

    cax = plotfuncs.plot_radar_timeseries(hampdata.radar, fig, axs[0, 0])[1]
    axs[0, 0].set_title("  Timeseries", fontsize=18, loc="left")

    plotfuncs.plot_radar_histogram(hampdata.radar, axs[0, 1])
    axs[0, 1].set_ylabel("")
    axs[0, 1].set_title("Histogram", fontsize=18)

    plotfuncs.plot_column_water_vapour_timeseries(
        hampdata["CWV"]["IWV"], axs[1, 0], target_cwv=48
    )

    plotfuncs.beautify_axes(axs.flatten())
    plotfuncs.beautify_colorbar_axes(cax)
    axs[1, 1].remove()

    fig.tight_layout()

    if savefigparams != []:
        save_figure(fig, savefigparams)

    return fig, axs


def add_itcz_mask(ax, xtime, itcz_mask, cbar=True):
    colors = ["red", "gold", "green"]  # Red, Green, Blue
    cmap = LinearSegmentedColormap.from_list("three_color_cmap", colors)
    levels = [-0.5, 0.5, 1.5, 2.5]

    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 2)
    xx, yy = np.meshgrid(xtime, y)
    z = np.array([itcz_mask, itcz_mask])

    cont = ax.contourf(
        xx,
        yy,
        z,
        levels=levels,
        cmap=cmap,
        alpha=0.2,
    )
    clab = "ITCZ Mask"
    if cbar:
        cbar = fig.colorbar(cont, ax=ax, label=clab, shrink=0.8)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Outside", "Transition", "Inside"])


def interpolate_radiometer_mask_to_radar_mask(itcz_mask, hampdata):
    """returns mask for radar time dimension interpolated
    from mask with radiometer (CWV) time dimension"""
    ds_mask1 = xr.Dataset(
        {
            "itcz_mask": xr.DataArray(
                itcz_mask, dims=hampdata["CWV"].dims, coords=hampdata["CWV"].coords
            )
        }
    )
    ds_mask2 = ds_mask1.interp(time=hampdata.radar.time)

    return ds_mask2.itcz_mask


# %% Plot CWV and radar with ITCZ mask
savefig_format = "png"
savename = Path(cfg["paths"]["saveplts"]) / "radar_column_water_path.png"
dpi = 64

fig, axes = plot_radar_cwv_timeseries(hampdata, figsize=(12, 6), savefigparams=[])
axes[1, 0].legend(loc="upper right", frameon=False)
itcz_mask_1 = itczfuncs.identify_itcz_crossings(hampdata["CWV"]["IWV"])
add_itcz_mask(axes[1, 0], hampdata["CWV"].time, itcz_mask_1)
itcz_mask_2 = interpolate_radiometer_mask_to_radar_mask(itcz_mask_1, hampdata)
add_itcz_mask(axes[0, 0], hampdata.radar.time, itcz_mask_2, cbar=False)
save_figure(fig, savefigparams=[savefig_format, savename, dpi])

# %% Plot radar timeseries and histogram for ecah masked area
savefig_format = "png"
savename = Path(cfg["paths"]["saveplts"]) / "radar_selected.png"
dpi = 64

fig, axs = plt.subplots(
    nrows=3, ncols=2, figsize=(12, 6), width_ratios=[18, 7], sharey="row", sharex="col"
)

time_radar = hampdata.radar.time
height_km = hampdata.radar.height / 1e3  # [km]
signal = plotfuncs.filter_radar_signal(hampdata.radar.dBZg, threshold=-30)  # [dBZ]

nrep = len(height_km)
itcz_mask_signal = np.repeat(itcz_mask_2, nrep)
itcz_mask_signal = np.reshape(itcz_mask_signal.values, [len(time_radar), nrep])

mask_values = {0: "Outside", 1: "Transition", 2: "Inside"}

for a in mask_values.keys():
    ax0, ax1 = axs[a, 0], axs[a, 1]

    signal_plt = np.where(itcz_mask_signal == a, signal, np.nan)

    ax1.set_title(f"{mask_values[a]}", loc="left")
    cax = plotfuncs.plot_radardata_timeseries(
        time_radar, height_km, signal_plt.T, fig, ax0
    )[1]
    ax0.set_xlabel("")

    signal_range = [-30, 30]
    height_range = [0, np.nanmax(height_km)]
    signal_bins = 60
    height_bins = 100
    cmap = plotfuncs.get_greys_histogram_colourmap(signal_bins)
    plotfuncs.plot_radardata_histogram(
        time_radar,
        height_km,
        signal_plt,
        ax1,
        signal_range,
        height_range,
        height_bins,
        signal_bins,
        cmap,
    )
    axs[a, 1].set_ylabel("")

axs[2, 0].set_xlabel("UTC")
axs[2, 1].set_xlabel("Z /dBZe")
plotfuncs.beautify_axes(axs.flatten())
plotfuncs.beautify_colorbar_axes(cax)

fig.tight_layout()

save_figure(fig, savefigparams=[savefig_format, savename, dpi])
