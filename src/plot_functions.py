import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def beautify_axes(axes):
    if not isinstance(type(axes), list):
        axes = [axes]

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)


def beautify_colorbar_axes(cax, xaxis=False):
    if xaxis:
        cax.ax.tick_params(axis="x", which="major", labelsize=12)
        cax.ax.xaxis.label.set_size(15)
    else:
        cax.ax.tick_params(axis="y", which="major", labelsize=12)
        cax.ax.yaxis.label.set_size(15)
    cax.ax.tick_params(labelsize=15)


def filter_radar_signal(dBZg, threshold=-30):
    return dBZg.where(dBZg >= threshold)  # [dBZ]


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
            bbox_to_anchor=(1.05, 0.5),
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
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)

    ax.set_ylabel("TB / K")


def plot_column_water_vapour_timeseries(ds, ax, target_cwv=None):
    """
    Plot column water vapour retrieval data (from KV band radiometer)

    Parameters
    ----------
    ds : xr.Dataset
        Level1 radiometer dataset for column water vapour retrieval
    ax : matplotlib.axes.Axes
        Axes to plot on.
    target_cwv : float
        value to CWV [mm] to mark on plot, e.g. 48mm contour
    """

    ds.plot.line(ax=ax, x="time", color="k")

    if target_cwv:
        # Plot where CWV equals the target value (within 0.01%)
        target_indices = np.where(
            np.isclose(ds.values, target_cwv, atol=1e-4 * target_cwv)
        )
        if ds[target_indices].any():
            ds[target_indices].plot.scatter(
                ax=ax, x="time", color="dodgerblue", marker="x"
            )

        ax.axhline(
            target_cwv,
            color="dodgerblue",
            linestyle="--",
            linewidth=1.0,
            label=f"CWV={target_cwv}mm",
        )

    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
    ax.set_ylabel("CWV / mm")


def plot_radar_timeseries(ds, fig, ax, cax=None, cmap="YlGnBu"):
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
        time, height = ds.time, ds.height / 1e3  # [UTC], [km]
        signal = filter_radar_signal(ds.dBZg, threshold=-30).T  # [dBZ]
        plot_radardata_timeseries(time, height, signal, fig, ax, cax=cax, cmap=cmap)


def plot_radardata_timeseries(time, height, signal, fig, ax, cax=None, cmap="YlGnBu"):
    """you may want to filter_radar_signal before calling this function"""
    pcol = ax.pcolormesh(
        time,
        height,
        signal,
        cmap=cmap,
        vmin=-30,
        vmax=30,
    )

    clab, extend, shrink = "Z /dBZe", "max", 0.8
    if cax:
        cax = fig.colorbar(pcol, cax=cax, label=clab, extend=extend, shrink=shrink)
    else:
        cax = fig.colorbar(pcol, ax=ax, label=clab, extend=extend, shrink=shrink)

    # get nicely formatting xticklabels
    stride = len(time) // 4
    xticks = time[::stride]
    xticklabs = [f"{t.hour:02d}:{t.minute:02d}" for t in pd.to_datetime(xticks)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabs)

    ax.set_xlabel("UTC")
    ax.set_ylabel("Height / km")

    return ax, cax, pcol


def plot_radar_histogram(
    ds_radar,
    ax,
    signal_range=[-30, 30],
    height_range=[],
    signal_bins=60,
    height_bins=100,
    cmap=None,
):
    if height_range == []:
        height_range = [0.0, ds_radar.height.max()]

    if not cmap:
        cmap = plt.get_cmap("Greys")
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "Sampled_Greys", cmap(np.linspace(0.15, 1.0, signal_bins))
        )
        cmap.set_under("white")
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    plot_radardata_histogram(
        ds_radar.time,
        ds_radar.height,
        ds_radar.dBZg,
        ax,
        signal_range,
        height_range,
        height_bins,
        signal_bins,
        cmap,
    )


def plot_radardata_histogram(
    time,
    height,
    signal,
    ax,
    signal_range,
    height_range,
    height_bins,
    signal_bins,
    cmap,
):
    # get data in correct format for 2D histogram
    height = np.meshgrid(height, time)[0].flatten() / 1e3  # [km]
    signal = filter_radar_signal(signal, threshold=-30).values.flatten()  # [dBZ]

    # remove nan data
    height = height[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]

    # set histogram parameters
    bins = [signal_bins, height_bins]
    cmap.set_under("white")

    # plot 2D histogram
    hist, xbins, ybins, im = ax.hist2d(
        signal,
        height,
        range=[signal_range, height_range],
        bins=bins,
        cmap=cmap,
        vmin=signal[signal > 0].min(),
    )

    ax.set_xlabel("Z /dBZe")
    ax.set_ylabel("Height / km")

    return ax, hist, xbins, ybins
