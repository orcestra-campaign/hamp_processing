import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

# import matplotlib
# matplotlib.use("webagg")


def plot_dropsonde(ds_extrap, ds_loc):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    ds_extrap["ta"].plot(y="gpsalt", ax=axes[0, 0], label="Temperature", color="red")
    ds_loc["ta"].plot(y="gpsalt", ax=axes[0, 0], label="Temperature", color="blue")
    ds_extrap["q"].plot(
        y="gpsalt", ax=axes[0, 1], label="Specific humidity", color="red"
    )
    ds_loc["q"].plot(y="gpsalt", ax=axes[0, 1], label="Specific humidity", color="blue")
    ds_extrap["p"].plot(y="gpsalt", ax=axes[0, 2], label="Pressure", color="red")
    ds_loc["p"].plot(y="gpsalt", ax=axes[0, 2], label="Pressure", color="blue")
    ds_extrap["alt"].plot(y="gpsalt", ax=axes[1, 0], label="Altitude", color="red")
    ds_loc["alt"].plot(y="gpsalt", ax=axes[1, 0], label="Altitude", color="blue")
    ds_loc["u"].plot(y="gpsalt", ax=axes[1, 1], label="u", color="blue")
    ds_loc["v"].plot(y="gpsalt", ax=axes[1, 2], label="v", color="blue")
    axes[0, 0].set_xlabel("Temperature / K")
    axes[0, 1].set_xlabel("Specific humidity")
    axes[0, 2].set_xlabel("Pressure / hPa")
    axes[1, 0].set_xlabel("Altitude / m")
    axes[1, 1].set_xlabel("u / m/s")
    axes[1, 2].set_xlabel("v / m/s")

    for ax in axes.flatten():
        ax.set_title("")
        ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("GPS altitude / m")
    axes[0, 1].set_ylabel("GPS altitude / m")
    plt.show()


def plot_TB_comparison(TB_arts, TB_hamp, sonde_id):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(TB_hamp.index, TB_hamp, color="blue", marker="o", label="HAMP")
    ax.scatter(TB_arts.index, TB_arts, color="red", marker="x", label="ARTS")
    ax.set_xlabel("Brightness temperature / K")
    ax.set_xticks(TB_arts.index)
    ax.set_xticklabels(TB_arts.index, rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Frequency / GHz")
    ax.set_title(f"{sonde_id}")
    ax.legend()
    plt.show()


def testplot_hamp(
    ds_radar,
    ds_radiometers,
    ds_iwv,
    ground_filter=True,
    roll_filter=True,
    calibration_filter=True,
    amplifier_faults=True,
):
    plt.close("all")
    fig, axes = plt.subplots(
        7,
        2,
        figsize=(15, 20),
        sharex="col",
        height_ratios=[3, 1, 1, 1, 1, 1, 1],
        width_ratios=[10, 1],
    )
    data = ds_radar["dBZg"]
    if ground_filter:
        data = data.where(ds_radar["mask_ground_return"])
    if roll_filter:
        data = data.where(ds_radar["mask_roll"])
    if calibration_filter:
        data = data.where(ds_radar["mask_calibration"])
    if amplifier_faults:
        ds_radiometers = ds_radiometers.where(ds_radiometers["mask_amplifier_fault"])

    col = data.plot.pcolormesh(
        ax=axes[0, 0],
        x="time",
        y="height",
        cmap="YlGnBu",
        vmin=-30,
        vmax=30,
        add_colorbar=False,
    )
    cb = fig.colorbar(col, cax=axes[0, 1], orientation="vertical", shrink=0.5)
    cb.set_label("dBZg")

    # Define frequencies for each subplot
    frequencies_list = [
        [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.4],
        [50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.0],
        [90],
        [120.15, 121.05, 122.95, 127.25],
        [183.91, 184.81, 185.81, 186.81, 188.31, 190.81],
    ]

    # Create a colormap
    cmap = get_cmap("viridis")  # Adjust the range according to your frequencies

    for i, frequencies in enumerate(frequencies_list):
        for f in frequencies:
            norm = Normalize(vmin=min(frequencies), vmax=max(frequencies))
            color = cmap(norm(f))
            ds_radiometers["TBs"].sel(frequency=[f]).plot(
                ax=axes[i + 1, 0], x="time", label=f, color=color
            )
        handles, labels = axes[i + 1, 0].get_legend_handles_labels()
        axes[i + 1, 1].legend(handles=handles, labels=labels)
        axes[i + 1, 1].axis("off")
        axes[i + 1, 0].set_title("")
        axes[i + 1, 0].set_ylabel("TB / K")
        axes[i + 1, 0].set_xlabel("")
        axes[i + 1, 0].spines[["top", "right"]].set_visible(False)

    ds_iwv["IWV"].plot(ax=axes[6, 0], x="time", color="k", label="IWV")
    axes[6, 1].remove()
    axes[6, 0].set_title("")
    axes[6, 0].set_ylabel("IWV / kg m-2")
    axes[0, 0].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Height / m")
    axes[6, 0].set_xlabel("Time")
    fig.tight_layout()

    return fig


def define_module(radio):
    if (radio == "K") | (radio == "V"):
        module = "KV"
    elif radio == "183":
        module = "183"
    elif radio == "90":
        module = "11990"
    elif radio == "119":
        module = "11990"
    else:
        raise ValueError("Invalid radiometer frequency")
    return module


def plot_radiometers(
    ds_radiometers,
    ds_radiometers_raw,
):
    plt.close("all")
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(12, 7),
        sharex="col",
        width_ratios=[10, 1],
    )

    freqs = {
        "K": slice(22, 32),
        "V": slice(50, 58),
        "183": slice(183, 191),
        "119": slice(120, 128),
        "90": 90,
    }
    for i, radio in enumerate(freqs.keys()):
        ds_radiometers["TBs"].sel(frequency=freqs[radio]).plot.line(
            ax=axes[i, 0], x="time", add_legend=False
        )
        if radio != "90":
            handles = list(axes[i, 0].get_lines())
            labels = ds_radiometers.sel(frequency=freqs[radio]).frequency.values
            axes[i, 1].legend(handles, labels)
        axes[i, 1].axis("off")
        axes[i, 0].set_ylabel("TB / K")
        axes[i, 0].set_xlabel("")
        axes[i, 0].set_title(f"{radio}")
        axes[i, 0].spines[["top"]].set_visible(False)
        ax2 = axes[i, 0].twinx()
        ax2.spines[["top"]].set_visible(False)
        ax2.plot(
            ds_radiometers_raw[define_module(radio)].time,
            ds_radiometers_raw[define_module(radio)].sel(frequency=freqs[radio])[
                "gain"
            ],
            color="grey",
        )
        ax2.set_ylabel("Gain")

    axes[4, 0].set_xlabel("Time")
    fig.tight_layout()
    plt.show()

    return fig


def plot_regression(regression_coeffs, TB_arts, TB_hamp, date):
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    TB_hamp = TB_hamp[TB_hamp.index.date == date]
    TB_arts = TB_arts[TB_arts.index.date == date]
    for i, freq in enumerate(TB_arts.columns):
        x = np.linspace(
            np.nanmin(TB_hamp[freq].values), np.nanmax(TB_hamp[freq].values), 100
        )
        y = (
            regression_coeffs.loc[date, (freq, "slope")] * x
            + regression_coeffs.loc[date, (freq, "intercept")]
        )
        axes.flatten()[i].scatter(
            TB_hamp[freq], TB_arts[freq], color="blue", marker="o", s=1
        )
        axes.flatten()[i].plot(x, y, color="red")
        axes.flatten()[i].set_title(f"{freq} GHz")

    for ax in axes[:, 0]:
        ax.set_ylabel("ARTS TB [K]")
    for ax in axes[-1, :]:
        ax.set_xlabel("HAMP TB [K]")
    fig.tight_layout()
