import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_radiometer_timeseries(ds, ax, is_90=False):
    """
    Plot radiometer data for all frequencies in dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """

    if is_90:
        ds.plot.line(ax=ax, x="time", color="k")
        ax.legend(
            handles=ax.lines,
            labels=["90 GHz"],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
    else:
        frequencies = ds.frequency.values
        norm = mcolors.Normalize(vmin=frequencies.min(), vmax=frequencies.max())
        cmap = plt.colormaps.get_cmap("viridis")

        for freq in frequencies:
            color = cmap(norm(freq))
            ds.sel(frequency=freq).plot.line(
                ax=ax, x="time", color=color, label=f"{freq:.2f} GHz"
            )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set_ylabel("TB / K")


def plot_radar_timeseries(ds, fig, ax, cax, cmap="plasma"):
    """WIP 15:41 UTC"""

    # check if radar data is available
    if ds.dBZg.size == 0:
        ax.text(
            0.5,
            0.5,
            "No radar data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        pcol = ax.pcolormesh(
            ds.time,
            ds.height / 1e3,
            ds.dBZg.where(ds.dBZg > -25).T,
            cmap=cmap,
            vmin=-25,
            vmax=25,
        )
        if cax:
            fig.colorbar(pcol, cax=cax, label="Reflectivity /dBZe", extend="max")
        else:
            fig.colorbar(pcol, ax=ax, label="Reflectivity /dBZe", extend="max")

    ax.set_xlabel("Time")
    ax.set_ylabel("Height / km")


def plot_radar_histogram(ds, ax, signal_range=[], height_range=[], cmap="magma_r"):
    # get data in correct format for 2D histogram
    height = np.meshgrid(ds.height, ds.time)[0].flatten() / 1e3  # [km]
    signal = ds.dBZg.where(ds.dBZg > -25).values.flatten()  # [dBZ]

    # remove nan data
    height = height[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]

    # set histogram parameters
    if height_range == []:
        height_range = [0.0, height.max()]
    if signal_range == []:
        signal_range = [signal.min(), signal.max()]

    # plot 2D histogram
    ax.hist2d(
        signal,
        height,
        range=[signal_range, height_range],
        bins=[len(ds.height), 60],
        cmap=cmap,
    )

    ax.set_xlabel("reflectivity, Z /dBZe")
    ax.set_ylabel("Height / km")
